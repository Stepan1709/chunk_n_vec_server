from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from fastapi import FastAPI

from app.clients.embedding import EmbeddingClient
from app.main import create_app

BASE_URL = "http://embeddings.test"


@pytest.fixture
def mock_embeddings() -> AsyncIterator[respx.MockRouter]:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        yield router  # type: ignore[misc]


@pytest.fixture
async def app(mock_embeddings: respx.MockRouter) -> AsyncIterator[FastAPI]:
    application = create_app()
    async with httpx.AsyncClient(timeout=5) as http_client:
        application.state.embedding_client = EmbeddingClient(http_client, BASE_URL, "test-model", 2)
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
