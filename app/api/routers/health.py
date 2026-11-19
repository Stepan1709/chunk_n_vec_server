"""Служебные эндпоинты: информация о сервисе, liveness и health."""

from fastapi import APIRouter

from app.api.deps import EmbeddingClientDep, SettingsDep
from app.schemas.health import HealthResponse

router = APIRouter(tags=["service"])


@router.get("/")
async def root() -> dict[str, object]:
    return {
        "service": "Chunking and Vectorization Service",
        "version": "2.0.0",
        "description": "Разбивает текст на чанки и получает для них эмбеддинги",
        "endpoints": {
            "/": "Информация о сервисе",
            "/live": "Liveness-проба (внешние сервисы не проверяет)",
            "/health": "Состояние сервиса и сервиса эмбеддингов",
            "/process": "Чанкинг и векторизация текста",
        },
    }


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health", response_model=HealthResponse)
async def health(client: EmbeddingClientDep, settings: SettingsDep) -> HealthResponse:
    available = await client.is_available()
    return HealthResponse(
        status="healthy" if available else "degraded",
        embedding_service=settings.embedding_base_url,
        embedding_service_status="available" if available else "unavailable",
        config={
            "server_port": settings.server_port,
            "embedding_model": settings.embedding_model,
            "default_max_chunk_size": settings.default_max_chunk_size,
            "default_overlap": settings.default_overlap,
            "chunking_strategy": "fixed_with_overlap",
            "max_concurrent_embedding_requests": settings.max_concurrent_embedding_requests,
        },
    )
