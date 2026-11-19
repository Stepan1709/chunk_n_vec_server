"""Зависимости FastAPI: доступ к объектам, созданным в lifespan."""

from typing import Annotated

from fastapi import Depends, Request

from app.clients.embedding import EmbeddingClient
from app.core.config import Settings, get_settings
from app.services.vectorization import VectorizationService


def get_embedding_client(request: Request) -> EmbeddingClient:
    client: EmbeddingClient = request.app.state.embedding_client
    return client


EmbeddingClientDep = Annotated[EmbeddingClient, Depends(get_embedding_client)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_vectorization_service(client: EmbeddingClientDep) -> VectorizationService:
    return VectorizationService(client)


VectorizationServiceDep = Annotated[VectorizationService, Depends(get_vectorization_service)]
