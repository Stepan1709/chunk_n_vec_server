"""Запуск сервиса: python -m app."""

import uvicorn

from app.core.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
    )
