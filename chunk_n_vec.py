import asyncio
import logging
import sys
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

import config

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Модели данных для API
class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Текст для обработки")
    max_chunk_size: int = Field(4000, description="Максимальный размер чанка в символах (без overlap)", ge=100,
                                le=10000)
    overlap: int = Field(500, description="Размер перекрытия между чанками в символах", ge=0, le=2000)


class ChunkEmbedding(BaseModel):
    embedding: List[float]
    chunk_text: str


class EmbeddingResponse(BaseModel):
    chunks: List[ChunkEmbedding]
    total_chunks: int
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    embedding_service: str
    embedding_service_status: str
    config: Dict[str, Any]


# Глобальные переменные
client: Optional[httpx.AsyncClient] = None


def chunk_text(text: str, max_chunk_size: int = 4000, overlap: int = 500) -> List[str]:
    """
    Разбиение текста на чанки фиксированного размера с перекрытием.

    Логика:
    - Чанк 1 содержит символы [0:max_chunk_size + overlap]
    - Чанк 2 содержит символы [max_chunk_size:2*max_chunk_size + overlap]
    - И т.д.

    Каждый чанк (кроме последнего в файле) включает overlap символов из следующего чанка.
    Если последний чанк меньше overlap символов, он не сохраняется (т.к. уже включен в предпоследний).

    Args:
        text: Текст для разбиения
        max_chunk_size: Максимальный размер чанка без overlap (по умолчанию 4000)
        overlap: Размер перекрытия между чанками (по умолчанию 500)

    Returns:
        Список чанков текста
    """
    logger.info(f"Chunking text with max_chunk_size={max_chunk_size}, overlap={overlap}")
    logger.info(f"Total text length: {len(text)}")

    if overlap >= max_chunk_size:
        error_msg = f"Overlap ({overlap}) must be less than max_chunk_size ({max_chunk_size})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    chunk_size_with_overlap = max_chunk_size + overlap
    chunk_texts = []

    # Вычисляем количество чанков
    total_length = len(text)

    if total_length <= chunk_size_with_overlap:
        # Текст помещается в один чанк
        chunk_texts.append(text)
        logger.info(f"Text fits in single chunk of size {total_length}")
        return chunk_texts

    # Разбиваем текст на чанки с перекрытием
    position = 0
    while position < total_length:
        # Определяем конец текущего чанка
        end_pos = min(position + chunk_size_with_overlap, total_length)

        # Получаем чанк
        chunk = text[position:end_pos]

        # Проверяем, не является ли это последним маленьким чанком
        if position + max_chunk_size >= total_length:
            # Это последний чанк
            if len(chunk) <= overlap:
                # Последний чанк слишком маленький и уже включен в предпоследний
                logger.info(f"Skipping last small chunk of size {len(chunk)} (already included in previous overlap)")
                break
            else:
                # Последний чанк достаточно большой
                chunk_texts.append(chunk)
                logger.info(f"Added last chunk: size={len(chunk)}, range=[{position}:{end_pos}]")
                break

        chunk_texts.append(chunk)
        logger.debug(f"Added chunk: size={len(chunk)}, range=[{position}:{end_pos}]")

        # Перемещаем позицию на начало следующего чанка
        position += max_chunk_size

    # Логируем статистику
    if chunk_texts:
        sizes = [len(chunk) for chunk in chunk_texts]
        min_size = min(sizes)
        max_size = max(sizes)
        avg_size = sum(sizes) / len(sizes)

        logger.info(f"Chunking completed - Total chunks: {len(chunk_texts)}, "
                    f"Min size: {min_size}, Max size: {max_size}, Avg size: {avg_size:.2f}")
    else:
        logger.warning("No chunks created")

    return chunk_texts


async def get_embedding(text: str, client: httpx.AsyncClient, url: str, model: str) -> List[float]:
    """
    Получение эмбеддинга для текста через удаленный сервис

    Args:
        text: Текст для векторизации
        client: HTTP клиент
        url: URL сервиса эмбеддингов
        model: Название модели

    Returns:
        Вектор эмбеддинга
    """
    try:
        payload = {
            "model": model,
            "input": text
        }

        logger.debug(f"Sending embedding request for text of length {len(text)}")

        response = await client.post(
            f"{url}/v1/embeddings",
            json=payload,
            timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()
        embedding = data["data"][0]["embedding"]

        logger.debug(f"Received embedding vector of dimension {len(embedding)}")
        return embedding

    except httpx.TimeoutException:
        logger.error(f"Timeout while getting embedding for text: {text[:100]}...")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Embedding service timeout"
        )
    except httpx.RequestError as e:
        logger.error(f"Request error while getting embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error while getting embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding processing error: {str(e)}"
        )


async def process_chunks_parallel(chunks: List[str], client: httpx.AsyncClient,
                                  url: str, model: str) -> List[ChunkEmbedding]:
    """
    Параллельная векторизация всех чанков

    Args:
        chunks: Список чанков текста
        client: HTTP клиент
        url: URL сервиса эмбеддингов
        model: Название модели

    Returns:
        Список объектов ChunkEmbedding
    """
    tasks = []
    for chunk in chunks:
        task = get_embedding(chunk, client, url, model)
        tasks.append(task)

    embeddings = await asyncio.gather(*tasks)

    results = []
    for chunk, embedding in zip(chunks, embeddings):
        results.append(ChunkEmbedding(
            embedding=embedding,
            chunk_text=chunk
        ))

    logger.info(f"Successfully vectorized {len(results)} chunks")
    return results


async def check_embedding_service_health(url: str) -> bool:
    """
    Проверка доступности сервиса эмбеддингов

    Args:
        url: URL сервиса эмбеддингов

    Returns:
        True если сервис доступен, иначе False
    """
    try:
        async with httpx.AsyncClient() as client:
            # Проверяем корневой эндпоинт
            response = await client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return True

            # Если /health нет, пробуем получить модели
            response = await client.get(f"{url}/v1/models", timeout=5.0)
            return response.status_code == 200

    except Exception as e:
        logger.warning(f"Embedding service health check failed: {str(e)}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global client

    # Инициализация HTTP клиента
    logger.info("Initializing HTTP client...")
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.REQUEST_TIMEOUT),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )

    logger.info(f"Starting chunking and vectorization service on port {config.SERVER_PORT}")
    logger.info(f"Using embedding service at: {config.VEC_URL}")
    logger.info(f"Using embedding model: {config.EMBEDDING_MODEL}")

    yield

    # Закрытие HTTP клиента
    logger.info("Shutting down HTTP client...")
    await client.aclose()
    logger.info("Service stopped")


# Создание FastAPI приложения
app = FastAPI(
    title="Chunking and Vectorization Service",
    description="Microservice for text chunking and embedding vectorization",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Корневой эндпоинт с информацией о сервисе"""
    logger.info("Root endpoint accessed")
    return {
        "service": "Chunking and Vectorization Service",
        "version": "1.0.0",
        "description": "Processes text into chunks and generates embeddings",
        "endpoints": {
            "/": "Service information",
            "/health": "Health check with embedding service status",
            "/process": "Process text into chunks with embeddings"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса и доступности эмбеддинг сервиса"""
    logger.info("Health check endpoint accessed")

    # Проверяем статус эмбеддинг сервиса
    embedding_status = await check_embedding_service_health(config.VEC_URL)

    health_status = HealthResponse(
        status="healthy" if embedding_status else "degraded",
        embedding_service=config.VEC_URL,
        embedding_service_status="available" if embedding_status else "unavailable",
        config={
            "server_port": config.SERVER_PORT,
            "embedding_model": config.EMBEDDING_MODEL,
            "default_max_chunk_size": 4000,
            "default_overlap": 500,
            "chunking_strategy": "fixed_with_overlap"
        }
    )

    logger.info(f"Health check result: {health_status.status}, "
                f"Embedding service: {health_status.embedding_service_status}")

    return health_status


@app.post("/process", response_model=EmbeddingResponse)
async def process_text(request: EmbeddingRequest):
    """
    Основной эндпоинт для обработки текста:
    1. Разбиение на чанки с перекрытием
    2. Векторизация чанков
    """
    start_time = time.time()

    logger.info(f"Processing request - text_length: {len(request.text)}, "
                f"max_chunk_size: {request.max_chunk_size}, "
                f"overlap: {request.overlap}")

    # 1. Разбиение на чанки
    logger.info("Step 1: Chunking text")

    try:
        chunks = chunk_text(
            request.text,
            request.max_chunk_size,
            request.overlap
        )
    except ValueError as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during chunking: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking failed: {str(e)}"
        )

    if not chunks:
        logger.warning("No chunks were created from the text")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks were created from the provided text"
        )

    # 2. Векторизация чанков
    logger.info(f"Step 2: Vectorizing {len(chunks)} chunks")

    try:
        results = await process_chunks_parallel(
            chunks,
            client,
            config.VEC_URL,
            config.EMBEDDING_MODEL
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during vectorization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vectorization failed: {str(e)}"
        )

    processing_time = time.time() - start_time

    logger.info(f"Request completed successfully - total_chunks: {len(results)}, "
                f"processing_time: {processing_time:.3f}s")

    return EmbeddingResponse(
        chunks=results,
        total_chunks=len(results),
        processing_time=processing_time
    )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info",
        access_log=True
    )