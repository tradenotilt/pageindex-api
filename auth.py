from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config import get_settings

settings = get_settings()
if not settings.app_api_key.strip():
    raise RuntimeError("APP_API_KEY не задан")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Проверяет API-ключ для защиты всех эндпоинтов."""
    if api_key is None or api_key != settings.app_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )
    return settings.app_api_key
