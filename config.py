"""Конфигурация сервиса. Все параметры задаются переменными окружения."""
import os

VEC_URL = os.getenv("VEC_URL", "http://localhost:8500").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_CONCURRENT_EMBEDDING_REQUESTS = int(os.getenv("MAX_CONCURRENT_EMBEDDING_REQUESTS", "5"))

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8998"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Значения по умолчанию для параметров чанкинга
DEFAULT_MAX_CHUNK_SIZE = 4000
DEFAULT_OVERLAP = 500
