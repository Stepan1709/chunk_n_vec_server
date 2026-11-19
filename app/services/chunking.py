"""Разбиение текста на чанки фиксированного размера с перекрытием."""

import logging

from app.core.exceptions import InvalidChunkingParamsError

logger = logging.getLogger(__name__)


def chunk_text(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    """
    Режет текст на чанки: чанк i содержит символы [i*max_chunk_size : (i+1)*max_chunk_size + overlap].

    Текст, помещающийся в max_chunk_size + overlap, возвращается одним чанком.
    Последний хвост не длиннее overlap не сохраняется: он уже входит в предыдущий чанк.
    """
    if overlap >= max_chunk_size:
        raise InvalidChunkingParamsError(
            f"overlap ({overlap}) должен быть меньше max_chunk_size ({max_chunk_size})"
        )

    window = max_chunk_size + overlap
    total = len(text)
    if total <= window:
        return [text]

    chunks: list[str] = []
    position = 0
    while position < total:
        chunk = text[position : position + window]
        if position + max_chunk_size >= total:
            # Последний чанк: хвост, целиком лежащий в перекрытии, уже учтён
            if len(chunk) > overlap:
                chunks.append(chunk)
            break
        chunks.append(chunk)
        position += max_chunk_size

    logger.info("Текст длиной %d разбит на %d чанков", total, len(chunks))
    return chunks
