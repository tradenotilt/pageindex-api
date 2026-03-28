from __future__ import annotations

import asyncio
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
        self.toc_check_page_num = int(os.getenv("PAGEINDEX_TOC_CHECK_PAGES", "20"))
        self.pdf_chunk_size = int(os.getenv("PAGEINDEX_PDF_CHUNK_SIZE", "1800"))
        self.pdf_chunk_overlap = int(os.getenv("PAGEINDEX_PDF_CHUNK_OVERLAP", "200"))
        self.pdf_chunk_threshold = int(os.getenv("PAGEINDEX_PDF_CHUNK_THRESHOLD", "1200"))

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

        normalized = self._normalize_result(result, source_path)
        saved_path = Path(self.settings.data_dir) / f"{doc_id}.json"
        saved_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(saved_path)

    def _process_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """Индексирует PDF. Сначала пытается PageIndex, затем Markdown-мост, затем плоский текст."""
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
                return self._normalize_result(result, pdf_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("PageIndex не нашел оглавление (%s). Включаю Markdown-мост...", exc)
        else:
            logger.warning("PageIndex не установлен. Включаю Markdown-мост...")

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
            return normalized

        except Exception as md_exc:
            logger.error("Markdown-мост не справился (%s). Перехожу на базовое извлечение.", md_exc)

        # 3. Базовый резервный план (плоский текст, если всё сломалось)
        return self._process_pdf_fallback(pdf_path)

    def _should_use_pdf_fallback(self, exc: Exception) -> bool:
        """Определяет, нужно ли переключаться на резервную обработку PDF (Оставлено для совместимости)."""
        message = str(exc).lower()
        return "toc_detected" in message or "json" in message or "table of content" in message

    def _process_pdf_fallback(self, pdf_path: Path) -> dict[str, Any]:
        """Резервно извлекает текст PDF без PageIndex."""
        page_texts = self._extract_pdf_page_texts(pdf_path)
        structure: list[dict[str, Any]] = []

        for page_number, text in page_texts:
            page_node_id = f"page_{page_number:04d}"
            page_node: dict[str, Any] = {
                "title": f"Страница {page_number}",
                "node_id": page_node_id,
                "page_index": page_number,
                "source_type": "pdf_page",
                "text": text,
                "nodes": [],
            }

            if len(text) >= self.pdf_chunk_threshold:
                page_node["nodes"] = self._build_chunk_nodes(
                    text=text,
                    parent_id=page_node_id,
                    page_index=page_number,
                    source_type="pdf_chunk",
                )
            structure.append(page_node)

        return {
            "doc_name": pdf_path.name,
            "doc_description": "PDF обработан через резервный режим без PageIndex",
            "structure": structure,
        }

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
                text = (page.get_text("text") or "").strip()
                if not text:
                    blocks = page.get_text("blocks")
                    text = "\n".join(
                        block[4].strip()
                        for block in blocks
                        if isinstance(block, (list, tuple)) and len(block) > 4 and str(block[4]).strip()
                    ).strip()
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