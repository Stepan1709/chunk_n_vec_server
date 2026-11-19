"""Доменные исключения и их преобразование в HTTP-ответы."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Базовое исключение сервиса."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Внутренняя ошибка сервиса"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.detail
        super().__init__(self.detail)


class InvalidChunkingParamsError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


class EmptyChunksError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    detail = "Из переданного текста не получилось ни одного чанка"


class EmbeddingServiceTimeoutError(AppError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    detail = "Сервис эмбеддингов не ответил вовремя"


class EmbeddingServiceUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Сервис эмбеддингов недоступен"


class EmbeddingServiceResponseError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Сервис эмбеддингов вернул некорректный ответ"


async def app_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
