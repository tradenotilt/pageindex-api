from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import openai
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from auth import get_api_key
from config import Settings, get_settings
from task_manager import TaskManager

logger = logging.getLogger(__name__)
settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class SearchRequest(BaseModel):
    doc_id: str
    query: str

class AgentRequest(BaseModel):
    query: str

def ensure_storage() -> None:
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.results_dir).mkdir(parents=True, exist_ok=True)

    registry_path = Path(settings.registry_file)
    if not registry_path.exists():
        registry_path.write_text("{}", encoding="utf-8")

    tasks_path = Path(settings.tasks_file)
    if not tasks_path.exists():
        tasks_path.write_text("{}", encoding="utf-8")

def load_registry() -> dict:
    registry_path = Path(settings.registry_file)
    if not registry_path.exists():
        return {}
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("Реестр документов повреждён")
        return {}
    return data if isinstance(data, dict) else {}

def save_registry(data: dict) -> None:
    registry_path = Path(settings.registry_file)
    registry_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ДЕРЕВА ---
def get_skeleton(nodes, max_preview_length=300):
    """Создает скелет документа с текстовыми превью для каждого узла."""
    if isinstance(nodes, dict):
        nodes = [nodes]
        
    skeleton = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
            
        new_n = {k: v for k, v in n.items() if k not in ['text', 'nodes']}
        
        # Добавляем краткое превью текста для понимания содержимого
        text = n.get("text", "")
        if text and len(text) > max_preview_length:
            preview = text[:max_preview_length] + "..."
        else:
            preview = text
        
        if preview.strip():
            new_n["text_preview"] = preview.strip()
        
        if 'nodes' in n and n['nodes']:
            new_n['nodes'] = get_skeleton(n['nodes'], max_preview_length)
        skeleton.append(new_n)
    return skeleton

def get_node_map(nodes, node_map):
    if isinstance(nodes, dict):
        nodes = [nodes]
        
    for n in nodes:
        if not isinstance(n, dict):
            continue
            
        if 'node_id' in n:
            node_map[n['node_id']] = n
        if 'nodes' in n and n['nodes']:
            get_node_map(n['nodes'], node_map)
    return node_map

def get_node_text_with_children(node: dict) -> str:
    """Рекурсивно собирает текст узла и его потомков без дублирования страниц PDF."""
    if not isinstance(node, dict):
        return ""

    children = node.get("nodes", [])
    child_texts = []
    if isinstance(children, list) and children:
        for child in children:
            child_text = get_node_text_with_children(child)
            if child_text.strip():
                child_texts.append(child_text.strip())

    text = (node.get("text", "") or "").strip()
    source_type = str(node.get("source_type") or "").lower()

    if child_texts:
        if source_type == "pdf_page":
            return "\n\n".join(child_texts).strip()
        if text:
            return "\n\n".join([text, *child_texts]).strip()
        return "\n\n".join(child_texts).strip()

    return text

def extract_document_nodes(tree_data: object) -> list[dict]:
    if isinstance(tree_data, dict) and "result" in tree_data:
        tree_data = tree_data["result"]

    if isinstance(tree_data, list):
        return [node for node in tree_data if isinstance(node, dict)]

    if isinstance(tree_data, dict):
        if "node_id" in tree_data:
            return [tree_data]
            
        structure = tree_data.get("structure")
        if isinstance(structure, list):
            return [node for node in structure if isinstance(node, dict)]

        nodes = tree_data.get("nodes")
        if isinstance(nodes, list):
            return [node for node in nodes if isinstance(node, dict)]

    raise HTTPException(status_code=500, detail="Неверный формат индекса документа")

def get_openai_client() -> openai.OpenAI:
    if not settings.has_openai_key:
        raise HTTPException(status_code=500, detail="CHATGPT_API_KEY не задан")
    return openai.OpenAI(api_key=settings.chatgpt_api_key)


def extract_message_content(message: object) -> str:
    content = getattr(message, "content", None)
    if not content:
        raise HTTPException(status_code=500, detail="OpenAI вернул пустой ответ")
    return content


def parse_json_response(raw_text: str, error_message: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = re.sub(r"^json\s*", "", text.strip(), flags=re.IGNORECASE)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    raise HTTPException(status_code=500, detail=error_message)


def collect_leaf_nodes(nodes: list[dict]) -> list[dict]:
    leaf_nodes: list[dict] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        children = node.get("nodes")
        if isinstance(children, list) and children:
            leaf_nodes.extend(collect_leaf_nodes(children))
        else:
            leaf_nodes.append(node)
    return leaf_nodes


def score_node_for_query(node: dict, query: str) -> int:
    """Ключевое слово-скоринг для определения релевантности узла запросу."""
    text = " ".join(
        str(part)
        for part in [
            node.get("title", ""),
            node.get("text", ""),
            node.get("summary", ""),
        ]
        if part
    ).lower()
    query_terms = [term for term in re.findall(r"[\wа-яА-ЯёЁ]+", query.lower()) if len(term) > 2]
    if not query_terms:
        return 0
    return sum(1 for term in query_terms if term in text)


def score_node_for_query_semantic(node: dict, query: str, client: openai.OpenAI) -> float:
    """Семантический скоринг с использованием embeddings для более точного определения релевантности."""
    text = " ".join(
        str(part)
        for part in [
            node.get("title", ""),
            node.get("text", ""),
            node.get("summary", ""),
        ]
        if part
    )
    
    if not text.strip():
        return 0.0
    
    try:
        # Ограничиваем длину текста для экономии токенов
        text_for_embedding = text[:1000] if len(text) > 1000 else text
        
        # Получаем embeddings для запроса и текста узла
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
        
        text_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=text_for_embedding
        ).data[0].embedding
        
        # Вычисляем косинусное сходство
        import numpy as np
        query_array = np.array(query_embedding)
        text_array = np.array(text_embedding)
        
        similarity = np.dot(query_array, text_array) / (
            np.linalg.norm(query_array) * np.linalg.norm(text_array)
        )
        
        return float(similarity)
    except Exception as exc:
        # Fallback к ключевое слово-скорингу при ошибке
        logger.warning("Ошибка семантического скоринга: %s", exc)
        return float(score_node_for_query(node, query))


def build_fallback_context(nodes: list[dict], query: str, limit: int = 5, use_semantic: bool = False, client: openai.OpenAI | None = None) -> str:
    """Создает контекст из наиболее релевантных узлов.
    
    Args:
        nodes: Список узлов документа
        query: Поисковый запрос
        limit: Максимальное количество узлов для включения
        use_semantic: Использовать ли семантический скоринг
        client: OpenAI клиент для семантического скоринга
    """
    leaf_nodes = collect_leaf_nodes(nodes)
    
    # Выбираем метод скоринга
    if use_semantic and client:
        ranked_nodes = sorted(
            leaf_nodes,
            key=lambda node: score_node_for_query_semantic(node, query, client),
            reverse=True,
        )
    else:
        ranked_nodes = sorted(
            leaf_nodes,
            key=lambda node: (
                score_node_for_query(node, query),
                len(str(node.get("text", "") or "")),
            ),
            reverse=True,
        )

    # Фильтруем узлы с положительным скором
    if use_semantic and client:
        selected = [node for node in ranked_nodes if score_node_for_query_semantic(node, query, client) > 0.3][:limit]
    else:
        selected = [node for node in ranked_nodes if score_node_for_query(node, query) > 0][:limit]
    
    if not selected:
        selected = ranked_nodes[:limit]

    chunks = []
    for node in selected:
        text = get_node_text_with_children(node).strip()
        if not text:
            continue
        title = node.get("title", "Без названия")
        chunks.append(f"Раздел: {title}\nТекст: {text}")

    return "\n\n".join(chunks)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage()
    task_manager = TaskManager(settings)
    await task_manager.load_tasks()
    app.state.task_manager = task_manager
    yield

app = FastAPI(lifespan=lifespan, dependencies=[Depends(get_api_key)])

# ==========================================
# ЭНДПОИНТЫ API
# ==========================================

@app.post("/upload", status_code=202)
async def upload_document(file: UploadFile = File(...)):
    """1. Загрузка файла (Асинхронная очередь) - Только файл, без ручного описания"""
    task_manager: TaskManager = app.state.task_manager
    file_bytes = await file.read()
    safe_filename = file.filename or "document"

    is_valid, error_message = task_manager.file_handler.validate_file(safe_filename, len(file_bytes))
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    file_path = Path(settings.data_dir) / f"{doc_id}_{safe_filename}"
    file_path.write_bytes(file_bytes)

    file_type = task_manager.file_handler.get_file_type(safe_filename)
    if not file_type:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

    job_id = await task_manager.create_task(
        doc_id=doc_id,
        filename=safe_filename,
        description="Генерация авто-описания...", # Временная заглушка
        file_type=file_type,
        file_path=str(file_path),
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "doc_id": doc_id,
        "message": "Документ добавлен в очередь на обработку",
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Удаляет документ и все связанные с ним данные (JSON, записи в реестре и задачах).
    """
    task_manager: TaskManager = app.state.task_manager
    success = await asyncio.to_thread(task_manager.delete_document, doc_id)
    
    if success:
        return {"status": "success", "message": f"Документ {doc_id} и все его данные успешно удалены."}
    else:
        raise HTTPException(status_code=404, detail="Документ не найден или уже удален.")

@app.get("/tasks/{job_id}")
async def get_task_status(job_id: str):
    task_manager: TaskManager = app.state.task_manager
    task = await task_manager.get_task_status(job_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return {"job_id": job_id, **task}

@app.get("/tasks")
async def list_tasks():
    task_manager: TaskManager = app.state.task_manager
    return {"tasks": await task_manager.get_all_tasks()}

@app.get("/registry")
async def list_registry():
    return load_registry()

@app.post("/search")
async def search_doc(request: SearchRequest):
    file_path = Path(settings.data_dir) / f"{request.doc_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Документ не найден")

    with file_path.open("r", encoding="utf-8") as file:
        tree_data = json.load(file)

    client = get_openai_client()
    nodes = extract_document_nodes(tree_data)
    skeleton = get_skeleton(nodes)
    
    search_prompt = (
        "Найди узлы в структуре, где может быть ответ на вопрос.\n"
        "У каждого узла есть текстовое превью (text_preview) для понимания содержимого.\n"
        f"Вопрос: {request.query}\n"
        f"Структура: {json.dumps(skeleton, ensure_ascii=False)}\n"
        'Верни строго JSON: {"thinking": "почему", "node_list": ["id1", "id2"]}'
    )

    try:
        response_search = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": search_prompt}],
            temperature=0.1,
        )
        raw_json = extract_message_content(response_search.choices[0].message)
        search_result = parse_json_response(
            raw_json,
            "OpenAI вернул неверный формат поиска по документу",
        )
        target_nodes = search_result.get("node_list", [])
        if not isinstance(target_nodes, list):
            target_nodes = []

        node_map: dict[str, dict] = {}
        get_node_map(nodes, node_map)

        relevant_blocks = []
        for node_id in target_nodes:
            node = node_map.get(str(node_id))
            if not node:
                continue
            text = get_node_text_with_children(node).strip()
            if not text:
                continue
            title = node.get("title", "Без названия")
            relevant_blocks.append(f"Раздел: {title}\nТекст: {text}")

        relevant_content = "\n\n".join(relevant_blocks)

        # Улучшенный fallback с семантическим поиском
        if not relevant_content.strip():
            relevant_content = build_fallback_context(nodes, request.query, use_semantic=True, client=client)

        # Если семантический поиск не помог, пробуем ключевое слово-скоринг
        if not relevant_content.strip():
            relevant_content = build_fallback_context(nodes, request.query, use_semantic=False)

        if not relevant_content.strip():
            relevant_content = (
                "Не удалось определить релевантные фрагменты. "
                "Изучи структуру документа и ответь по доступному контексту."
            )

        answer_prompt = (
            "Ответь на вопрос по контексту.\n"
            f"Вопрос: {request.query}\n"
            f"Контекст: {relevant_content}"
        )
        response_answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.1,
        )
        return {"status": "success", "answer": response_answer.choices[0].message.content}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка поиска по документу %s", request.doc_id)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/vision")
async def vision_rag(file: UploadFile = File(...), query: str = Form(...)):
    client = get_openai_client()
    file_bytes = await file.read()
    mime_type = (file.content_type or "").lower()

    filename = file.filename or ""
    if "pdf" in mime_type or filename.lower().endswith(".pdf"):
        try:
            import fitz  # type: ignore

            doc = fitz.open(stream=file_bytes, filetype="pdf")  # type: ignore[attr-defined]
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_bytes = pix.tobytes("jpeg")
            mime_type = "image/jpeg"
        except ImportError:
            raise HTTPException(status_code=500, detail="Для чтения PDF установите PyMuPDF (pymupdf)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка конвертации PDF: {str(e)}")
    else:
        image_bytes = file_bytes

    if not mime_type.startswith("image/"):
        if file.filename:
            suffix = Path(file.filename).suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
            mime_type = mime_map.get(suffix, mime_type)

    if not mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Vision принимает только изображения с MIME type image/* или PDF")

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Изучи это изображение и ответь: {query}"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                    ],
                }
            ],
        )
        return {"status": "success", "answer": response.choices[0].message.content}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка vision-запроса")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/ask_agent")
async def ask_agent(request: AgentRequest):
    registry = load_registry()
    if not registry:
        return {"answer": "База документов пуста. Загрузите файлы."}

    client = get_openai_client()
    registry_items = [
        {
            "doc_id": doc_id,
            "filename": info.get("filename"),
            "file_type": info.get("file_type"),
            "description": info.get("description"),
        }
        for doc_id, info in registry.items()
        if isinstance(info, dict)
    ]

    route_prompt = (
        f"Доступные документы: {json.dumps(registry_items, ensure_ascii=False)}\n"
        f"Вопрос пользователя: {request.query}\n"
        "Какой doc_id лучше всего подходит для ответа?\n"
        'Верни строго JSON: {"doc_id": "doc_...", "reason": "почему"} '
        'или {"doc_id": null} если нет подходящего.'
    )

    try:
        response_route = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": route_prompt}],
            temperature=0.1,
        )
        raw_route = extract_message_content(response_route.choices[0].message)
        route_result = parse_json_response(
            raw_route,
            "OpenAI вернул неверный формат маршрута",
        )

        target_doc_id = route_result.get("doc_id")
        if not target_doc_id:
            return {"answer": "К сожалению, в моей базе нет подходящего документа для ответа на этот вопрос."}

        search_request = SearchRequest(doc_id=str(target_doc_id), query=request.query)
        search_result = await search_doc(search_request)

        registry_entry = registry.get(str(target_doc_id))
        filename = registry_entry.get("filename") if isinstance(registry_entry, dict) else None

        return {
            "status": "success",
            "used_doc_id": str(target_doc_id),
            "filename": filename,
            "answer": search_result["answer"],
        }
    except KeyError as exc:
        logger.exception("Выбранный документ отсутствует в реестре")
        raise HTTPException(status_code=404, detail="Выбранный документ не найден") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка агентного поиска")
        raise HTTPException(status_code=500, detail=str(exc))