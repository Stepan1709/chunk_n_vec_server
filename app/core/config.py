"""Конфигурация сервиса. Все параметры задаются переменными окружения или файлом .env."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vec_url: str = "http://localhost:8500"
    embedding_model: str = "BAAI/bge-m3"
    request_timeout: float = Field(120.0, gt=0)
    max_concurrent_embedding_requests: int = Field(5, ge=1)

    server_host: str = "0.0.0.0"
    server_port: int = Field(8998, ge=1, le=65535)
    log_level: str = "INFO"

    # Значения по умолчанию для параметров чанкинга
    default_max_chunk_size: int = 4000
    default_overlap: int = 500

    @property
    def embedding_base_url(self) -> str:
        return self.vec_url.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()
