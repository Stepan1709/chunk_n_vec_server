"""Схемы служебных эндпоинтов."""

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    embedding_service: str
    embedding_service_status: str
    config: dict[str, Any]
