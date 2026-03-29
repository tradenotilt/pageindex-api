from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from PyPDF2 import PdfReader

from config import Settings

import importlib

PAGEINDEX_IMPORT_ERROR: Exception | None = None
MD_TO_TREE_IMPORT_ERROR: Exception | None = None


def _resolve_pageindex_callable(module_paths: list[str], attribute: str) -> tuple[Any | None, Exception | None]:
    """Пытается найти нужную функцию PageIndex в разных модулях."""
    last_error: Exception | None = None

    for module_path in module_paths:
        try:
            module = importlib.import_module(module_path)
            candidate = getattr(module, attribute)
            return candidate, None
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc

    return None, last_error


page_index, PAGEINDEX_IMPORT_ERROR = _resolve_pageindex_callable(
    ["pageindex", "pageindex.page_index"],
    "page_index",
)
md_to_tree, MD_TO_TREE_IMPORT_ERROR = _resolve_pageindex_callable(
    ["pageindex.page_index_md", "pageindex.md_to_tree"],
    "md_to_tree",
)

logger = logging.getLogger(__name__)


class FileHandler:
    """Валидирует входные файлы и индексирует их через PageIndex."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.supported_formats = {
            ".pdf": "pdf",
            ".doc": "doc",
            ".docx": "docx",
            ".md": "md",
        }
        self.pageindex_model = os.getenv("PAGEINDEX_MODEL", "gpt-4o-2024-11-20")
        # Увеличиваем проверку оглавления до 50 страниц для сложных документов
        self.toc_check_page_num = int(os.getenv("PAGEINDEX_TOC_CHECK_PAGES", "50"))
        self.pdf_chunk_size = int(os.getenv("PAGEINDEX_PDF_CHUNK_SIZE", "1800"))
        self.pdf_chunk_overlap = int(os.getenv("PAGEINDEX_PDF_CHUNK_OVERLAP", "200"))
        self.pdf_chunk_threshold = int(os.getenv("PAGEINDEX_PDF_CHUNK_THRESHOLD", "1200"))
        # Добавляем флаг для принудительного использования улучшенного парсинга
        self.force_enhanced_parsing = os.getenv("FORCE_ENHANCED_PARSING", "false").lower() == "true"
        # Модель для генерации keywords (по умолчанию: gpt-4o)
        self.keywords_model = os.getenv("KEYWORDS_MODEL", "gpt-4o")
        # Модель для Vision OCR (по умолчанию: gpt-4o)
        self.vision_model = os.getenv("VISION_MODEL", "gpt-4o")

    def get_file_type(self, filename: str) -> Optional[str]:
        """Определяет тип файла по расширению."""
        extension = Path(filename).suffix.lower()
        return self.supported_formats.get(extension)

    def validate_file(self, filename: str, size: int) -> tuple[bool, str]:
        """Проверяет формат и размер файла."""
        extension = Path(filename).suffix.lower()

        if extension not in self.settings.allowed_extensions:
            return False, f"Неподдерживаемый формат файла: {extension}"

        if size > self.settings.max_file_size:
            return False, (
                f"Размер файла превышает лимит {self.settings.max_file_size} байт"
            )

        return True, ""

    async def process_file(self, file_path: str, doc_id: str) -> str:
        """Индексирует файл и сохраняет нормализованное дерево в data/."""
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        extension = source_path.suffix.lower()
        if extension == ".md":
            result = await asyncio.to_thread(self._process_markdown, source_path)
        elif extension in {".doc", ".docx"}:
            result = await asyncio.to_thread(self._process_word_document, source_path)
        elif extension == ".pdf":
            result = await asyncio.to_thread(self._process_pdf, source_path)
        else:
            raise ValueError(f"Неподдерживаемый формат для индексации: {extension}")

        # Для PDF добавляем описания изображений через Vision OCR
        if extension == ".pdf":
            images = self._extract_images_from_pdf(source_path)
            if images:
                logger.info(f"Найдено {len(images)} страниц с изображениями")
                # Добавляем описания изображений в структуру документа (асинхронно)
                result = await asyncio.to_thread(self._add_image_descriptions, result, images)

        normalized = self._normalize_result(result, source_path)
        
        # Генерируем индекс документа с keywords через LLM
        normalized = await self._generate_document_index(normalized, source_path)
        
        saved_path = Path(self.settings.data_dir) / f"{doc_id}.json"
        saved_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(saved_path)

    def _process_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """Индексирует PDF с улучшенной обработкой таблиц и структуры."""
        # Если включен принудительный улучшенный парсинг, пропускаем PageIndex
        if self.force_enhanced_parsing:
            logger.info("Принудительный улучшенный парсинг включен. Использую fallback с распознаванием таблиц.")
            return self._process_pdf_fallback(pdf_path)
        
        # 1. Попытка классического PageIndex (для документов с оглавлением)
        if page_index is not None:
            try:
                result = page_index(
                    doc=str(pdf_path),
                    model=self.pageindex_model,
                    toc_check_page_num=self.toc_check_page_num,
                    max_page_num_each_node=10,
                    max_token_num_each_node=20000,
                    if_add_node_id="yes",
                    if_add_node_summary="yes",
                    if_add_doc_description="yes",
                    if_add_node_text="yes",
                )
                
                # Проверяем качество результата PageIndex
                normalized = self._normalize_result(result, pdf_path)
                structure = normalized.get("structure", [])
                
                # Если структура пустая или слишком простая, используем улучшенный fallback
                if not structure or len(structure) < 3:
                    logger.warning("PageIndex вернул простую структуру. Использую улучшенный fallback с распознаванием таблиц.")
                    return self._process_pdf_fallback(pdf_path)
                
                # Добавляем распознавание таблиц к результату PageIndex
                tables = self._extract_tables_from_pdf(pdf_path)
                if tables:
                    logger.info(f"PageIndex сработал, но добавляю распознанные таблицы ({len(tables)} страниц)")
                    self._enrich_with_tables(normalized["structure"], tables)
                
                return normalized
                
            except Exception as exc:  # noqa: BLE001
                logger.warning("PageIndex не справился (%s). Включаю улучшенный fallback с распознаванием таблиц...", exc)
        else:
            logger.warning("PageIndex не установлен. Включаю улучшенный fallback с распознаванием таблиц...")

        # 2. Умный резервный режим через Markdown-мост
        try:
            import pymupdf4llm

            logger.info("Конвертирую PDF в Markdown для построения структуры...")
            md_text = pymupdf4llm.to_markdown(str(pdf_path))

            md_text = re.sub(
                r'^(\d+(?:\.\d+)*\.?)\s+([A-ZА-ЯЁ][^\n]{2,150})$',
                r'## \1 \2',
                md_text,
                flags=re.MULTILINE,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_md_path = Path(temp_dir) / f"{pdf_path.stem}.md"
                temp_md_path.write_text(md_text, encoding="utf-8")
                tree_result = self._process_markdown(temp_md_path)

            normalized = self._normalize_result(tree_result, pdf_path)
            normalized["doc_description"] = "PDF обработан через умный Markdown-мост (pymupdf4llm)"
            
            # Добавляем распознавание таблиц к Markdown-мосту
            tables = self._extract_tables_from_pdf(pdf_path)
            if tables:
                logger.info(f"Добавляю распознанные таблицы к Markdown-мосту ({len(tables)} страниц)")
                self._enrich_with_tables(normalized["structure"], tables)
            
            return normalized

        except Exception as md_exc:
            logger.error("Markdown-мост не справился (%s). Перехожу на улучшенный fallback.", md_exc)

        # 3. Улучшенный резервный план с распознаванием таблиц
        return self._process_pdf_fallback(pdf_path)

    def _enrich_with_tables(self, structure: list[dict[str, Any]], tables: dict[int, str]) -> None:
        """Обогащает существующую структуру распознанными таблицами."""
        for page_node in structure:
            if not isinstance(page_node, dict):
                continue
            
            page_index = page_node.get("page_index")
            if page_index in tables and tables[page_index]:
                page_node_id = page_node.get("node_id", f"page_{page_index:04d}")
                
                # Создаем узлы для таблиц
                table_nodes = self._create_table_nodes(
                    tables[page_index],
                    page_node_id,
                    page_index
                )
                
                # Добавляем таблицы в текст страницы
                current_text = page_node.get("text", "")
                page_node["text"] = current_text + tables[page_index]
                
                # Добавляем узлы таблиц
                if "nodes" not in page_node:
                    page_node["nodes"] = []
                page_node["nodes"].extend(table_nodes)

    def _should_use_pdf_fallback(self, exc: Exception) -> bool:
        """Определяет, нужно ли переключаться на резервную обработку PDF (Оставлено для совместимости)."""
        message = str(exc).lower()
        return "toc_detected" in message or "json" in message or "table of content" in message

    def _process_pdf_fallback(self, pdf_path: Path) -> dict[str, Any]:
        """Резервно извлекает текст PDF без PageIndex с распознаванием таблиц."""
        page_texts = self._extract_pdf_page_texts(pdf_path)
        tables = self._extract_tables_from_pdf(pdf_path)
        structure: list[dict[str, Any]] = []

        for page_number, text in page_texts:
            page_node_id = f"page_{page_number:04d}"
            page_text = text
            
            # Добавляем распознанные таблицы к тексту страницы
            if page_number in tables and tables[page_number]:
                page_text += tables[page_number]
                logger.info(f"Добавлены таблицы на страницу {page_number}")
            
            page_node: dict[str, Any] = {
                "title": f"Страница {page_number}",
                "node_id": page_node_id,
                "page_index": page_number,
                "source_type": "pdf_page",
                "text": page_text,
                "nodes": [],
            }

            # Создаем отдельные узлы для таблиц
            if page_number in tables and tables[page_number]:
                table_nodes = self._create_table_nodes(
                    tables[page_number],
                    page_node_id,
                    page_number
                )
                page_node["nodes"].extend(table_nodes)

            if len(page_text) >= self.pdf_chunk_threshold:
                chunk_nodes = self._build_chunk_nodes(
                    text=page_text,
                    parent_id=page_node_id,
                    page_index=page_number,
                    source_type="pdf_chunk",
                )
                page_node["nodes"].extend(chunk_nodes)
            
            structure.append(page_node)

        return {
            "doc_name": pdf_path.name,
            "doc_description": "PDF обработан через резервный режим с распознаванием таблиц",
            "structure": structure,
        }

    def _create_table_nodes(
        self,
        tables_text: str,
        parent_id: str,
        page_index: int
    ) -> list[dict[str, Any]]:
        """Создает отдельные узлы для распознанных таблиц."""
        nodes: list[dict[str, Any]] = []
        
        # Разделяем таблицы по маркеру ===
        table_sections = re.split(r'=== Таблица \d+ ===', tables_text)
        table_sections = [s.strip() for s in table_sections if s.strip()]
        
        for table_index, table_content in enumerate(table_sections, start=1):
            # Создаем краткое описание таблицы из первых строк
            lines = table_content.split('\n')
            description_lines = []
            for line in lines[:5]:  # Берем первые 5 строк для описания
                if line.strip() and not line.startswith('-'):
                    description_lines.append(line.strip())
            
            table_description = " | ".join(description_lines[:3])  # Краткое описание
            
            node: dict[str, Any] = {
                "title": f"Таблица {table_index} на странице {page_index}",
                "node_id": f"{parent_id}_table_{table_index:02d}",
                "text": table_content,
                "text_preview": table_description[:300] + "..." if len(table_description) > 300 else table_description,
                "nodes": [],
                "source_type": "pdf_table",
                "page_index": page_index,
                "table_index": table_index,
            }
            nodes.append(node)
        
        return nodes

    def _process_markdown(self, md_path: Path) -> dict[str, Any]:
        """Индексирует Markdown через официальный md_to_tree workflow."""
        if md_to_tree is None:
            raise RuntimeError(
                "Пакет pageindex не установлен. Добавьте зависимость в requirements.txt"
            )

        return asyncio.run(
            md_to_tree(
                md_path=str(md_path),
                model=self.pageindex_model,
                if_thinning=True,
                min_token_threshold=5000,
                if_add_node_summary="yes",
                summary_token_threshold=200,
                if_add_doc_description="yes",
                if_add_node_text="yes",
                if_add_node_id="yes",
            )
        )

    def _process_word_document(self, word_path: Path) -> dict[str, Any]:
        """Конвертирует DOC/DOCX в PDF и индексирует результат через PageIndex."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            converted_pdf = self._convert_to_pdf(word_path, temp_dir_path)
            return self._process_pdf(converted_pdf)

    def _convert_to_pdf(self, source_path: Path, output_dir: Path) -> Path:
        """Конвертирует Word-документ в PDF через LibreOffice."""
        command = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(source_path),
        ]

        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.settings.task_timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "LibreOffice (soffice) не найден. Для DOC/DOCX в Docker-образ нужно установить LibreOffice"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Ошибка конвертации Word-документа в PDF: {exc.stderr.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Таймаут конвертации Word-документа: {self.settings.task_timeout} секунд"
            ) from exc

        converted_pdf = output_dir / f"{source_path.stem}.pdf"
        if not converted_pdf.exists():
            raise RuntimeError("LibreOffice не создал PDF-файл для индексации")

        return converted_pdf

    async def _generate_document_index(self, result: dict[str, Any], source_path: Path) -> dict[str, Any]:
        """Генерирует индекс документа с автоматической генерацией keywords через LLM."""
        try:
            import openai
        except ImportError:
            logger.warning("OpenAI не установлен. Индекс документа не будет создан.")
            return result
        
        structure = result.get("structure", [])
        if not isinstance(structure, list):
            return result
        
        # Собираем информацию о секциях, таблицах и изображениях
        sections = []
        tables = []
        images = []
        total_chunks = 0
        
        def collect_nodes(nodes: list[dict[str, Any]], parent_section: dict[str, Any] | None = None) -> None:
            nonlocal total_chunks
            
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                
                source_type = node.get("source_type", "")
                node_id = node.get("node_id", "")
                title = node.get("title", "")
                text = node.get("text", "")
                page_index = node.get("page_index")
                
                # Собираем секции
                if source_type in ["pdf_leaf", "pdf_page"] and parent_section is None:
                    # Считаем количество чанков в этой секции
                    chunk_count = 0
                    children = node.get("nodes", [])
                    if isinstance(children, list):
                        for child in children:
                            if child.get("source_type") == "pdf_chunk":
                                chunk_count += 1
                    
                    sections.append({
                        "section_id": node_id,
                        "title": title,
                        "page_range": [page_index, page_index] if page_index else [],
                        "chunk_count": chunk_count,
                        "keywords": [],  # Будут заполнены через LLM
                        "summary": text[:200] + "..." if len(text) > 200 else text,
                    })
                    total_chunks += chunk_count
                
                # Собираем таблицы
                if source_type == "pdf_table":
                    tables.append({
                        "table_id": node_id,
                        "title": title,
                        "page": page_index,
                        "chunk_id": node_id,
                        "keywords": [],  # Будут заполнены через LLM
                        "summary": text[:200] + "..." if len(text) > 200 else text,
                    })
                
                # Собираем изображения
                if source_type == "pdf_images":
                    images.append({
                        "image_id": node_id,
                        "title": title,
                        "page": page_index,
                        "chunk_id": node_id,
                        "keywords": [],  # Будут заполнены через LLM
                        "summary": text[:200] + "..." if len(text) > 200 else text,
                    })
                
                # Рекурсивно обрабатываем дочерние узлы
                children = node.get("nodes")
                if isinstance(children, list):
                    collect_nodes(children, node if source_type in ["pdf_leaf", "pdf_page"] else parent_section)
        
        collect_nodes(structure)
        
        # Генерируем keywords для секций через LLM
        if sections:
            try:
                client = openai.OpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
                
                # Формируем промпт для генерации keywords
                sections_text = "\n\n".join([
                    f"Секция {i+1}: {s['title']}\nОписание: {s['summary']}"
                    for i, s in enumerate(sections[:10])  # Ограничиваем до 10 секций
                ])
                
                prompt = f"""
Для следующего документа сгенерируй ключевые слова (keywords) для каждой секции.
Ключевые слова должны быть на русском языке, разделены запятыми, содержать 3-5 слов.
Формат ответа: номер секции: ключевое слово 1, ключевое слово 2, ...

Документ: {source_path.name}
Описание: {result.get('doc_description', '')}

Секции:
{sections_text}
"""
                
                response = client.chat.completions.create(
                    model=self.keywords_model,
                    messages=[
                        {"role": "system", "content": "Ты эксперт по анализу документов. Генерируй точные ключевые слова для секций."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                )
                
                keywords_text = response.choices[0].message.content
                
                # Парсим ответ LLM
                for line in keywords_text.split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        try:
                            section_num = int(parts[0].strip()) - 1  # Конвертируем в индекс
                            if 0 <= section_num < len(sections):
                                keywords = [kw.strip() for kw in parts[1].split(',') if kw.strip()]
                                sections[section_num]["keywords"] = keywords
                        except (ValueError, IndexError):
                            pass
                
                logger.info(f"Сгенерированы keywords для {len(sections)} секций")
                
            except Exception as exc:
                logger.warning("Ошибка при генерации keywords для секций: %s", exc)
        
        # Генерируем keywords для таблиц через LLM
        if tables:
            try:
                client = openai.OpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
                
                tables_text = "\n\n".join([
                    f"Таблица {i+1}: {t['title']}\nОписание: {t['summary']}"
                    for i, t in enumerate(tables[:5])  # Ограничиваем до 5 таблиц
                ])
                
                prompt = f"""
Для следующего документа сгенерируй ключевые слова (keywords) для каждой таблицы.
Ключевые слова должны быть на русском языке, разделены запятыми, содержать 3-5 слов.
Формат ответа: номер таблицы: ключевое слово 1, ключевое слово 2, ...

Документ: {source_path.name}

Таблицы:
{tables_text}
"""
                
                response = client.chat.completions.create(
                    model=self.keywords_model,
                    messages=[
                        {"role": "system", "content": "Ты эксперт по анализу таблиц. Генерируй точные ключевые слова для таблиц."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                )
                
                keywords_text = response.choices[0].message.content
                
                # Парсим ответ LLM
                for line in keywords_text.split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        try:
                            table_num = int(parts[0].strip()) - 1  # Конвертируем в индекс
                            if 0 <= table_num < len(tables):
                                keywords = [kw.strip() for kw in parts[1].split(',') if kw.strip()]
                                tables[table_num]["keywords"] = keywords
                        except (ValueError, IndexError):
                            pass
                
                logger.info(f"Сгенерированы keywords для {len(tables)} таблиц")
                
            except Exception as exc:
                logger.warning("Ошибка при генерации keywords для таблиц: %s", exc)
        
        # Создаем индекс документа
        document_index = {
            "title": result.get("doc_name", source_path.name),
            "description": result.get("doc_description", ""),
            "total_pages": len([n for n in structure if n.get("page_index")]),
            "total_chunks": total_chunks,
            "sections": sections,
            "tables": tables,
            "images": images,
        }
        
        # Добавляем индекс в начало результата
        result["document_index"] = document_index
        
        logger.info(f"Создан индекс документа: {len(sections)} секций, {len(tables)} таблиц, {len(images)} изображений")
        
        return result

    def _normalize_result(self, result: Any, source_path: Path) -> dict[str, Any]:
        """Приводит результат PageIndex к единому формату для дальнейшего поиска."""
        normalized: dict[str, Any]

        if isinstance(result, dict):
            if "structure" in result:
                normalized = result
            elif "result" in result and isinstance(result["result"], dict):
                normalized = result["result"]
            else:
                normalized = {
                    "doc_name": source_path.name,
                    "structure": [],
                    "raw_result": result,
                }
        elif isinstance(result, list):
            normalized = {
                "doc_name": source_path.name,
                "structure": result,
            }
        else:
            normalized = {
                "doc_name": source_path.name,
                "structure": [],
                "raw_result": result,
            }

        structure = normalized.get("structure")
        if isinstance(structure, list):
            normalized["structure"] = self._enrich_structure(structure, source_path)

        if "doc_name" not in normalized:
            normalized["doc_name"] = source_path.name

        return normalized

    def _enrich_structure(self, nodes: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
        """Добавляет чанк-уровень в PDF-структуру, чтобы повысить точность поиска."""
        enriched: list[dict[str, Any]] = []

        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue

            current = dict(node)
            children = current.get("nodes")
            if isinstance(children, list) and children:
                current["nodes"] = self._enrich_structure(children, source_path)
            else:
                text = str(current.get("text") or "").strip()
                if source_path.suffix.lower() == ".pdf" and len(text) >= self.pdf_chunk_threshold:
                    parent_id = str(current.get("node_id") or current.get("title") or f"node_{index}")
                    current["nodes"] = self._build_chunk_nodes(
                        text=text,
                        parent_id=parent_id,
                        page_index=current.get("page_index"),
                        source_type="pdf_chunk",
                    )
                    current["source_type"] = current.get("source_type") or "pdf_leaf"
            enriched.append(current)

        return enriched

    def _extract_pdf_page_texts(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Извлекает текст страниц PDF с приоритетом на PyMuPDF, затем PyPDF2."""
        try:
            import fitz  # type: ignore

            doc = fitz.open(str(pdf_path))
            page_texts: list[tuple[int, str]] = []
            for page_number, page in enumerate(doc, start=1):
                # Используем "text" с указанием кодировки для корректного извлечения русского текста
                text = (page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE) or "").strip()
                
                # Если текст пустой, пробуем извлечь через blocks
                if not text:
                    blocks = page.get_text("blocks", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                    text = "\n".join(
                        str(block[4]).strip()
                        for block in blocks
                        if isinstance(block, (list, tuple)) and len(block) > 4 and str(block[4]).strip()
                    ).strip()
                
                # Если все еще пустой, пробуем через dict
                if not text:
                    text_dict = page.get_text("dict")
                    if text_dict and "blocks" in text_dict:
                        text_lines = []
                        for block in text_dict["blocks"]:
                            if "lines" in block:
                                for line in block["lines"]:
                                    if "spans" in line:
                                        for span in line["spans"]:
                                            if "text" in span:
                                                text_lines.append(span["text"])
                        text = "\n".join(text_lines).strip()
                
                if text:
                    page_texts.append((page_number, text))
            return page_texts
        except ImportError:
            logger.info("PyMuPDF не установлен, использую PyPDF2 для извлечения текста PDF")
        except Exception as exc:  # noqa: BLE001
            logger.warning("PyMuPDF не смог извлечь текст PDF (%s), перехожу на PyPDF2", exc)

        reader = PdfReader(str(pdf_path))
        page_texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_texts.append((page_number, text))
        return page_texts

    def _extract_tables_from_pdf(self, pdf_path: Path) -> dict[int, str]:
        """Извлекает таблицы из PDF с помощью camelot."""
        try:
            import camelot
        except ImportError:
            logger.warning("camelot не установлен. Таблицы не будут распознаны.")
            return {}

        tables_dict: dict[int, str] = {}
        
        try:
            # Используем lattice mode для более точного распознавания таблиц
            tables = camelot.read_pdf(
                str(pdf_path),
                pages='all',
                flavor='lattice',
                strip_text='\n',
                edge_tol=500,
            )
            
            logger.info(f"Обнаружено {tables.n} таблиц в документе")
            
            for i, table in enumerate(tables):
                if table.df is not None and not table.df.empty:
                    page_num = table.page
                    # Преобразуем таблицу в читаемый формат
                    table_text = self._format_table_as_text(table.df)
                    
                    if page_num not in tables_dict:
                        tables_dict[page_num] = ""
                    
                    tables_dict[page_num] += f"\n\n=== Таблица {i + 1} ===\n{table_text}\n"
                    
        except Exception as exc:
            logger.warning("Ошибка при извлечении таблиц: %s", exc)
            
        return tables_dict

    def _format_table_as_text(self, df) -> str:
        """Форматирует DataFrame в текстовое представление таблицы."""
        if df.empty:
            return ""
        
        # Получаем заголовки
        headers = df.columns.tolist()
        
        # Форматируем каждую строку
        rows = []
        for _, row in df.iterrows():
            row_text = " | ".join(str(val) if val is not None else "" for val in row)
            rows.append(row_text)
        
        # Собираем таблицу
        header_text = " | ".join(str(h) for h in headers)
        separator = "-" * len(header_text)
        
        table_str = f"{header_text}\n{separator}\n"
        table_str += "\n".join(rows)
        
        return table_str

    def _extract_images_from_pdf(self, pdf_path: Path) -> dict[int, list[dict[str, Any]]]:
        """Извлекает изображения из PDF для описания через Vision OCR."""
        try:
            import fitz  # type: ignore
        except ImportError:
            logger.warning("PyMuPDF не установлен. Изображения не будут извлечены.")
            return {}
        
        images_dict: dict[int, list[dict[str, Any]]] = {}
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_number, page in enumerate(doc, start=1):
                page_images = []
                
                # Извлекаем изображения со страницы
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list, start=1):
                    try:
                        # Получаем изображение в формате base64
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        
                        # Проверяем, что изображение успешно извлечено
                        if not base_image or "image" not in base_image:
                            continue
                        
                        pix = base_image["image"]
                        
                        # Конвертируем в JPEG для экономии памяти
                        # Ограничиваем размер до 1024x1024, сохраняя пропорции
                        if hasattr(pix, 'width') and hasattr(pix, 'height'):
                            max_size = 1024
                            if pix.width > max_size or pix.height > max_size:
                                ratio = min(max_size / pix.width, max_size / pix.height)
                                new_width = int(pix.width * ratio)
                                new_height = int(pix.height * ratio)
                                pix = pix.resize(new_width, new_height)
                        
                        image_bytes = pix.tobytes("jpeg")
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        
                        image_info = {
                            "image_base64": image_base64,
                            "mime_type": "image/jpeg",
                            "index": img_index,
                            "width": pix.width if hasattr(pix, 'width') else 0,
                            "height": pix.height if hasattr(pix, 'height') else 0,
                        }
                        page_images.append(image_info)
                    except Exception as img_exc:
                        logger.warning(f"Ошибка при извлечении изображения {img_index} со страницы {page_number}: {img_exc}")
                        continue
                
                if page_images:
                    images_dict[page_number] = page_images
                    logger.info(f"Извлечено {len(page_images)} изображений со страницы {page_number}")
            
            doc.close()
            
        except Exception as exc:
            logger.warning("Ошибка при извлечении изображений из PDF: %s", exc)
        
        return images_dict

    async def _describe_images_with_vision(self, images: list[dict[str, Any]], query: str) -> str:
        """Описывает изображения с помощью Vision OCR."""
        if not images:
            return ""
        
        try:
            import openai
        except ImportError:
            logger.warning("OpenAI не установлен. Описания изображений не будут созданы.")
            return ""
        
        try:
            client = openai.OpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
            descriptions = []
            
            for img_info in images[:5]:  # Ограничиваем до 5 изображений для экономии
                prompt = f"Опиши это изображение в контексте вопроса: '{query}'. Укажи, что это за график или диаграмма, какие данные оно показывает."
                
                response = client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{img_info['mime_type']};base64,{img_info['image_base64']}"
                                    }
                                },
                            ],
                        }
                    ],
                    max_tokens=300,  # Ограничиваем длину описания
                )
                
                description = response.choices[0].message.content
                descriptions.append(f"Изображение {img_info['index']}: {description}")
            
            return "\n\n".join(descriptions)
            
        except Exception as exc:
            logger.warning("Ошибка при описании изображений через Vision: %s", exc)
            return ""

    def _add_image_descriptions(self, result: dict[str, Any], images: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
        """Добавляет описания изображений в структуру документа."""
        if not images:
            return result
        
        structure = result.get("structure", [])
        if not isinstance(structure, list):
            return result
        
        # Добавляем описания изображений к соответствующим страницам
        for page_node in structure:
            if not isinstance(page_node, dict):
                continue
            
            page_index = page_node.get("page_index")
            if page_index in images and images[page_index]:
                # Создаем отдельный узел для описаний изображений
                image_descriptions = "\n".join([
                    f"Изображение {img['index']}: {img['width']}x{img['height']}px"
                    for img in images[page_index]
                ])
                
                image_node = {
                    "title": f"Графики и изображения на странице {page_index}",
                    "node_id": f"page_{page_index:04d}_images",
                    "text": image_descriptions,
                    "text_preview": f"Графики и изображения ({len(images[page_index])} шт.)",
                    "nodes": [],
                    "source_type": "pdf_images",
                    "page_index": page_index,
                }
                
                # Добавляем узел с изображениями
                if "nodes" not in page_node:
                    page_node["nodes"] = []
                page_node["nodes"].append(image_node)
                
                # Также добавляем описания в текст страницы
                current_text = page_node.get("text", "")
                page_node["text"] = current_text + "\n\n" + image_descriptions
        
        return result

    def _build_chunk_nodes(
        self,
        text: str,
        parent_id: str,
        page_index: int | None = None,
        source_type: str = "pdf_chunk",
    ) -> list[dict[str, Any]]:
        """Разбивает длинный текст на устойчивые поисковые чанки."""
        chunks = self._split_text_into_chunks(text)
        nodes: list[dict[str, Any]] = []

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            node: dict[str, Any] = {
                "title": f"Фрагмент {chunk_index}",
                "node_id": f"{parent_id}_chunk_{chunk_index:04d}",
                "text": chunk_text,
                "nodes": [],
                "source_type": source_type,
                "chunk_index": chunk_index,
            }
            if page_index is not None:
                node["page_index"] = page_index
            nodes.append(node)

        return nodes

    def _split_text_into_chunks(self, text: str) -> list[str]:
        """Делит длинный текст на чанки по абзацам с небольшим overlap."""
        normalized_text = re.sub(r"\r\n?", "\n", text).strip()
        if not normalized_text:
            return []

        paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized_text) if part.strip()]
        if not paragraphs:
            paragraphs = [normalized_text]

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            candidates = [paragraph]
            if len(paragraph) > self.pdf_chunk_size:
                candidates = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?。！？])\s+", paragraph)
                    if sentence.strip()
                ] or [paragraph]

            for candidate in candidates:
                if not current:
                    current = candidate
                    continue

                if len(current) + len(candidate) + 2 <= self.pdf_chunk_size:
                    current = f"{current}\n\n{candidate}"
                    continue

                chunks.append(current)
                current = candidate

        if current:
            chunks.append(current)

        if self.pdf_chunk_overlap > 0 and len(chunks) > 1:
            overlapped: list[str] = [chunks[0]]
            for chunk in chunks[1:]:
                previous_tail = overlapped[-1][-self.pdf_chunk_overlap :].strip()
                if previous_tail:
                    overlapped.append(f"{previous_tail}\n\n{chunk}".strip())
                else:
                    overlapped.append(chunk)
            chunks = overlapped

        return chunks