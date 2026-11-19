"""Эндпоинт чанкинга и векторизации текста."""

from fastapi import APIRouter

from app.api.deps import VectorizationServiceDep
from app.schemas.process import ProcessRequest, ProcessResponse

router = APIRouter(tags=["process"])


@router.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest, service: VectorizationServiceDep) -> ProcessResponse:
    """Разбивает текст на чанки с перекрытием и векторизует каждый чанк."""
    return await service.process(request.text, request.max_chunk_size, request.overlap)
