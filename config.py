from functools import lru_cache
from typing import Any, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения, защищенные от ошибок парсинга."""

    chatgpt_api_key: str = Field(default="", alias="CHATGPT_API_KEY")
    app_api_key: str = Field(default="", alias="APP_API_KEY")
    max_concurrent_tasks: int = Field(default=10, alias="MAX_CONCURRENT_TASKS")
    task_timeout: int = Field(default=300, alias="TASK_TIMEOUT")
    max_file_size: int = Field(default=50 * 1024 * 1024, alias="MAX_FILE_SIZE")
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    results_dir: str = Field(default="./results", alias="RESULTS_DIR")
    registry_file: str = Field(default="./data/registry.json", alias="REGISTRY_FILE")
    tasks_file: str = Field(default="./data/tasks.json", alias="TASKS_FILE")
    pageindex_script: str = Field(default="run_pageindex.py", alias="PAGEINDEX_SCRIPT")

    allowed_extensions: Any = Field(
        default=".pdf,.doc,.docx,.md",
        alias="ALLOWED_EXTENSIONS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, value: Any) -> list[str]:
        """Нормализует расширения к формату '.ext'."""
        if value is None:
            return [".pdf", ".doc", ".docx", ".md"]

        if isinstance(value, list):
            raw_items = value
        else:
            text = str(value).strip()
            if not text:
                return [".pdf", ".doc", ".docx", ".md"]

            text = text.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
            raw_items = [part.strip() for part in text.split(",") if part.strip()]

        normalized: list[str] = []
        for item in raw_items:
            extension = str(item).strip().lower()
            if not extension:
                continue
            if not extension.startswith("."):
                extension = f".{extension}"
            normalized.append(extension)

        return normalized or [".pdf", ".doc", ".docx", ".md"]

    @property
    def has_openai_key(self) -> bool:
        """Проверяет, задан ли ключ OpenAI."""
        return bool(self.chatgpt_api_key.strip())

    @property
    def has_app_api_key(self) -> bool:
        """Проверяет, задан ли ключ для защиты API."""
        return bool(self.app_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает кэшированный экземпляр настроек."""
    return Settings()