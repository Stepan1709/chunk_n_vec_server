"""Точка входа: сборка FastAPI-приложения."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.routers import health, process
from app.clients.embedding import EmbeddingClient
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    ) as http_client:
        app.state.embedding_client = EmbeddingClient(
            http_client,
            settings.embedding_base_url,
            settings.embedding_model,
            settings.max_concurrent_embedding_requests,
        )
        logger.info(
            "Сервис запущен: эмбеддинги %s, модель %s",
            settings.embedding_base_url,
            settings.embedding_model,
        )
        yield
    logger.info("Сервис остановлен")


def create_app() -> FastAPI:
    setup_logging(get_settings().log_level)
    app = FastAPI(
        title="Chunking and Vectorization Service",
        description="Микросервис чанкинга текста и векторизации чанков",
        version="2.0.0",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(process.router)
    return app


app = create_app()
