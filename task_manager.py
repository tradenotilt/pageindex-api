from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import openai  # Добавили клиент OpenAI

from config import Settings
from file_handler import FileHandler

logger = logging.getLogger(__name__)


class TaskManager:
    """Управляет очередью задач обработки документов."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tasks: dict[str, dict[str, Any]] = {}
        self.processing_count = 0
        self.lock = asyncio.Lock()
        self.file_handler = FileHandler(settings)
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Создаёт нужные директории и файлы хранилища."""
        Path(self.settings.data_dir).mkdir(parents=True, exist_ok=True)

        tasks_path = Path(self.settings.tasks_file)
        if not tasks_path.exists():
            tasks_path.write_text("{}", encoding="utf-8")

        registry_path = Path(self.settings.registry_file)
        if not registry_path.exists():
            registry_path.write_text("{}", encoding="utf-8")

    def _utc_now(self) -> str:
        """Возвращает текущий UTC timestamp в ISO формате."""
        return datetime.now(timezone.utc).isoformat()

    async def load_tasks(self) -> None:
        """Загружает задачи из файла."""
        tasks_path = Path(self.settings.tasks_file)
        if tasks_path.exists():
            self.tasks = json.loads(tasks_path.read_text(encoding="utf-8"))

    async def save_tasks(self) -> None:
        """Сохраняет задачи в файл."""
        tasks_path = Path(self.settings.tasks_file)
        tasks_path.write_text(
            json.dumps(self.tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def create_task(
        self,
        doc_id: str,
        filename: str,
        description: str,
        file_type: str,
        file_path: str,
    ) -> str:
        """Создаёт новую задачу и возвращает job_id."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        self.tasks[job_id] = {
            "status": "pending",
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "file_path": file_path,
            "description": description,
            "created_at": self._utc_now(),
            "updated_at": self._utc_now(),
            "error": None,
        }
        await self.save_tasks()
        asyncio.create_task(self._try_process_task(job_id))
        return job_id

    async def get_task_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Возвращает статус задачи по job_id."""
        return self.tasks.get(job_id)

    async def get_all_tasks(self) -> list[dict[str, Any]]:
        """Возвращает список всех задач."""
        return list(self.tasks.values())

    async def _try_process_task(self, job_id: str) -> None:
        """Пытается запустить задачу, если есть свободный слот."""
        async with self.lock:
            if self.processing_count >= self.settings.max_concurrent_tasks:
                return

            task = self.tasks.get(job_id)
            if not task or task["status"] != "pending":
                return

            task["status"] = "processing"
            task["updated_at"] = self._utc_now()
            self.processing_count += 1
            await self.save_tasks()

        asyncio.create_task(self._process_task(job_id))

    async def _process_task(self, job_id: str) -> None:
        """Обрабатывает задачу в фоне."""
        task = self.tasks.get(job_id)
        if not task:
            return

        file_path = task["file_path"]
        doc_id = task["doc_id"]

        try:
            # Шаг 1: Парсим файл
            await self.file_handler.process_file(file_path, doc_id)
            
           # --- ШАГ 2: МАГИЯ АВТО-ОПИСАНИЯ ---
            final_description = task["description"]
            try:
                parsed_file_path = Path(self.settings.data_dir) / f"{doc_id}.json"
                if parsed_file_path.exists() and self.settings.has_openai_key:
                    with open(parsed_file_path, "r", encoding="utf-8") as f:
                        parsed_data = json.load(f)
                    
                    # Берем первые 8000 символов (достаточно для понимания сути)
                    sample_text = json.dumps(parsed_data, ensure_ascii=False)[:8000]
                    
                    client = openai.OpenAI(api_key=self.settings.chatgpt_api_key)
                    prompt = (
                        "Ты - умный архивариус базы данных. Составь краткое описание документа для поискового ИИ-агента. "
                        "Опиши суть, тип документа и название компании/проекта (если есть). "
                        "ЖЕСТКИЕ ПРАВИЛА: "
                        "1. Напиши ответ одним связным абзацем (1-2 предложения, до 200 символов). "
                        "2. ЗАПРЕЩЕНО использовать переносы строк (\\n), списки, маркеры или Markdown. "
                        "3. ЗАПРЕЩЕНО использовать вводные слова и префиксы вроде 'Тип документа:', 'Суть:', 'Название:'. "
                        "Пиши сразу по делу, например: 'Аналитический отчет компании Axenix об экономике и рисках внедрения ИИ-агентов.'\n\n"
                        f"Текст документа: {sample_text}"
                    )
                    
                    def get_summary():
                        return client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1
                        ).choices[0].message.content.strip()

                    final_description = await asyncio.to_thread(get_summary)
                    task["description"] = final_description # Обновляем описание в самой задаче!
                        
            except Exception as e:
                logger.error("Не удалось сгенерировать авто-описание для %s: %s", doc_id, e)

            # Шаг 3: Сохраняем в реестр
            registry = self._load_registry()
            registry[doc_id] = {
                "filename": task["filename"],
                "file_type": task["file_type"],
                "description": final_description,
                "uploaded_at": task["created_at"],
            }
            self._save_registry(registry)
            
            task["status"] = "completed"
            task["updated_at"] = self._utc_now()
            task["error"] = None
            self._remove_file(file_path)
            
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка обработки задачи %s", job_id)
            task["status"] = "failed"
            task["updated_at"] = self._utc_now()
            task["error"] = str(exc)
            self._remove_file(file_path)
        finally:
            async with self.lock:
                self.processing_count = max(0, self.processing_count - 1)
                await self.save_tasks()
            await self._process_next_task()

    async def _process_next_task(self) -> None:
        """Запускает следующую ожидающую задачу."""
        async with self.lock:
            if self.processing_count >= self.settings.max_concurrent_tasks:
                return

            for job_id, task in self.tasks.items():
                if task.get("status") == "pending":
                    asyncio.create_task(self._try_process_task(job_id))
                    break

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        """Загружает реестр документов."""
        registry_path = Path(self.settings.registry_file)
        if registry_path.exists():
            return json.loads(registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        """Сохраняет реестр документов."""
        registry_path = Path(self.settings.registry_file)
        registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _remove_file(self, file_path: str) -> None:
        """Удаляет временный файл, если он существует."""
        if os.path.exists(file_path):
            os.remove(file_path)
            
    def delete_document(self, doc_id: str) -> bool:
        """Удаляет все следы документа: файлы, записи в задачах и реестре."""
        deleted_something = False

        # 1. Удаляем физические файлы (оригинал и JSON)
        for p in Path(self.settings.data_dir).glob(f"{doc_id}*"):
            try:
                if p.is_file():
                    p.unlink()
                    deleted_something = True
            except Exception as e:
                logger.error("Ошибка при удалении файла %s: %s", p, e)

        # 2. Очищаем registry.json
        registry_path = Path(self.settings.registry_file)
        if registry_path.exists():
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            
            if doc_id in registry:
                del registry[doc_id]
                with open(registry_path, "w", encoding="utf-8") as f:
                    json.dump(registry, f, ensure_ascii=False, indent=2)
                deleted_something = True

        # 3. Очищаем tasks.json (находим задачу по doc_id)
        tasks_path = Path(self.settings.tasks_file)
        if tasks_path.exists():
            with open(tasks_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            
            job_to_delete = None
            for job_id, task_info in tasks.items():
                if task_info.get("doc_id") == doc_id:
                    job_to_delete = job_id
                    break
            
            if job_to_delete:
                del tasks[job_to_delete]
                with open(tasks_path, "w", encoding="utf-8") as f:
                    json.dump(tasks, f, ensure_ascii=False, indent=2)
                deleted_something = True

        return deleted_something