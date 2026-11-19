import pytest

from app.core.exceptions import InvalidChunkingParamsError
from app.services.chunking import chunk_text


def test_short_text_is_single_chunk() -> None:
    assert chunk_text("a" * 140, 100, 50) == ["a" * 140]


def test_chunks_overlap() -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(350))
    chunks = chunk_text(text, 100, 20)
    assert chunks[0] == text[0:120]
    assert chunks[1] == text[100:220]
    # перекрытие соседних чанков равно overlap
    assert chunks[0][-20:] == chunks[1][:20]


def test_tail_inside_overlap_is_dropped() -> None:
    # длина 210: чанки [0:120], [100:210]; хвост [200:210] целиком внутри второго чанка
    chunks = chunk_text("x" * 210, 100, 20)
    assert [len(c) for c in chunks] == [120, 110]


def test_last_chunk_longer_than_overlap_is_kept() -> None:
    chunks = chunk_text("x" * 300, 100, 20)
    assert [len(c) for c in chunks] == [120, 120, 100]


def test_overlap_not_less_than_chunk_size_raises() -> None:
    with pytest.raises(InvalidChunkingParamsError):
        chunk_text("x" * 500, 100, 100)
