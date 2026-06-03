import logging
from typing import Optional

# Настройки по умолчанию
DEFAULT_VEC_URL = "http://localhost:8500"  # URL по умолчанию
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_REQUEST_TIMEOUT = 120

# Порт сервера
SERVER_PORT = 8998
SERVER_HOST = "0.0.0.0"

# Пытаемся импортировать из secrets_app.py
try:
    from secrets_app import (
        VEC_URL as SECRETS_VEC_URL,
        EMBEDDING_MODEL as SECRETS_EMBEDDING_MODEL,
        REQUEST_TIMEOUT as SECRETS_REQUEST_TIMEOUT
    )

    # Используем значения из secrets_app.py если они есть
    VEC_URL = SECRETS_VEC_URL
    EMBEDDING_MODEL = SECRETS_EMBEDDING_MODEL
    REQUEST_TIMEOUT = SECRETS_REQUEST_TIMEOUT

    logging.info("Loaded configuration from secrets_app.py")

except ImportError:
    # Используем значения по умолчанию
    VEC_URL = DEFAULT_VEC_URL
    EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
    REQUEST_TIMEOUT = DEFAULT_REQUEST_TIMEOUT

    logging.warning("secrets_app.py not found, using default configuration")
except AttributeError as e:
    # Если в secrets_app.py не все переменные определены
    VEC_URL = getattr(__import__('secrets', fromlist=['VEC_URL']), 'VEC_URL', DEFAULT_VEC_URL)
    EMBEDDING_MODEL = getattr(__import__('secrets', fromlist=['EMBEDDING_MODEL']), 'EMBEDDING_MODEL',
                              DEFAULT_EMBEDDING_MODEL)
    REQUEST_TIMEOUT = getattr(__import__('secrets', fromlist=['REQUEST_TIMEOUT']), 'REQUEST_TIMEOUT',
                              DEFAULT_REQUEST_TIMEOUT)

    logging.warning("Some variables missing in secrets_app.py, using defaults for missing ones")