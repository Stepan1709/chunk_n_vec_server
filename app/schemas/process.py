"""Схемы запроса и ответа эндпоинта /process."""

from pydantic import BaseModel, Field

from app.core.config import get_settings

_settings = get_settings()


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для обработки")
    max_chunk_size: int = Field(
        _settings.default_max_chunk_size,
        ge=100,
        le=10000,
        description="Максимальный размер чанка в символах (без overlap)",
    )
    overlap: int = Field(
        _settings.default_overlap,
        ge=0,
        le=2000,
        description="Размер перекрытия между чанками в символах",
    )


class ChunkEmbedding(BaseModel):
    embedding: list[float]
    chunk_text: str


class ProcessResponse(BaseModel):
    chunks: list[ChunkEmbedding]
    total_chunks: int
    processing_time: float
