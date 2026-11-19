"""Клиент OpenAI-совместимого сервиса эмбеддингов (например, vLLM)."""

import asyncio
import logging

import httpx

from app.core.exceptions import (
    EmbeddingServiceResponseError,
    EmbeddingServiceTimeoutError,
    EmbeddingServiceUnavailableError,
)

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Получает эмбеддинги; число одновременных запросов ограничено семафором."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        model: str,
        max_concurrent_requests: int,
    ) -> None:
        self._http = http_client
        self._base_url = base_url
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def embed(self, text: str) -> list[float]:
        async with self._semaphore:
            try:
                response = await self._http.post(
                    f"{self._base_url}/v1/embeddings",
                    json={"model": self._model, "input": text},
                )
                response.raise_for_status()
                return [float(x) for x in response.json()["data"][0]["embedding"]]
            except httpx.TimeoutException as exc:
                logger.error("Таймаут при получении эмбеддинга (длина текста %d)", len(text))
                raise EmbeddingServiceTimeoutError() from exc
            except httpx.HTTPStatusError as exc:
                logger.error("Сервис эмбеддингов вернул %d", exc.response.status_code)
                raise EmbeddingServiceResponseError(
                    f"Сервис эмбеддингов вернул статус {exc.response.status_code}"
                ) from exc
            except httpx.RequestError as exc:
                logger.error("Ошибка запроса к сервису эмбеддингов: %r", exc)
                raise EmbeddingServiceUnavailableError() from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                logger.error("Неожиданный формат ответа сервиса эмбеддингов: %r", exc)
                raise EmbeddingServiceResponseError() from exc

    async def is_available(self) -> bool:
        """Проверяет доступность сервиса: сначала /health, затем /v1/models."""
        try:
            for path in ("/health", "/v1/models"):
                response = await self._http.get(f"{self._base_url}{path}", timeout=5.0)
                if response.status_code == 200:
                    return True
        except httpx.HTTPError as exc:
            logger.warning("Проверка сервиса эмбеддингов не удалась: %r", exc)
        return False
