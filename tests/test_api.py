import httpx
import respx


async def test_live(client: httpx.AsyncClient) -> None:
    response = await client.get("/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_process_returns_embeddings(
    client: httpx.AsyncClient, mock_embeddings: respx.MockRouter
) -> None:
    route = mock_embeddings.post("/v1/embeddings").respond(json={"data": [{"embedding": [0.1, 0.2]}]})

    response = await client.post("/process", json={"text": "x" * 250, "max_chunk_size": 100, "overlap": 20})

    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] == route.call_count == 3
    assert body["chunks"][0]["embedding"] == [0.1, 0.2]


async def test_process_invalid_overlap_returns_400(client: httpx.AsyncClient) -> None:
    response = await client.post("/process", json={"text": "hello", "max_chunk_size": 100, "overlap": 100})
    assert response.status_code == 400


async def test_process_empty_text_returns_422(client: httpx.AsyncClient) -> None:
    response = await client.post("/process", json={"text": ""})
    assert response.status_code == 422


async def test_embedding_service_error_returns_502(
    client: httpx.AsyncClient, mock_embeddings: respx.MockRouter
) -> None:
    mock_embeddings.post("/v1/embeddings").respond(status_code=500)
    response = await client.post("/process", json={"text": "hello"})
    assert response.status_code == 502


async def test_embedding_service_timeout_returns_504(
    client: httpx.AsyncClient, mock_embeddings: respx.MockRouter
) -> None:
    mock_embeddings.post("/v1/embeddings").mock(side_effect=httpx.ReadTimeout("timeout"))
    response = await client.post("/process", json={"text": "hello"})
    assert response.status_code == 504


async def test_embedding_service_down_returns_503(
    client: httpx.AsyncClient, mock_embeddings: respx.MockRouter
) -> None:
    mock_embeddings.post("/v1/embeddings").mock(side_effect=httpx.ConnectError("refused"))
    response = await client.post("/process", json={"text": "hello"})
    assert response.status_code == 503


async def test_health_degraded_when_service_unavailable(
    client: httpx.AsyncClient, mock_embeddings: respx.MockRouter
) -> None:
    mock_embeddings.get("/health").mock(side_effect=httpx.ConnectError("refused"))
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


async def test_health_healthy(client: httpx.AsyncClient, mock_embeddings: respx.MockRouter) -> None:
    mock_embeddings.get("/health").respond(200)
    response = await client.get("/health")
    assert response.json()["status"] == "healthy"
