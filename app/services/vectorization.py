"""Сценарий обработки текста: чанкинг и векторизация чанков."""

import asyncio
import logging
import time

from app.clients.embedding import EmbeddingClient
from app.core.exceptions import EmptyChunksError
from app.schemas.process import ChunkEmbedding, ProcessResponse
from app.services.chunking import chunk_text

logger = logging.getLogger(__name__)


class VectorizationService:
    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self._embedding_client = embedding_client

    async def process(self, text: str, max_chunk_size: int, overlap: int) -> ProcessResponse:
        started = time.perf_counter()

        chunks = chunk_text(text, max_chunk_size, overlap)
        if not chunks:
            raise EmptyChunksError()

        # Параллелизм ограничен семафором внутри клиента
        embeddings = await asyncio.gather(*(self._embedding_client.embed(c) for c in chunks))
        results = [
            ChunkEmbedding(embedding=embedding, chunk_text=chunk)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        elapsed = time.perf_counter() - started
        logger.info("Обработано чанков: %d за %.3f c", len(results), elapsed)
        return ProcessResponse(chunks=results, total_chunks=len(results), processing_time=elapsed)
