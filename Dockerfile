# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка только необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements.txt отдельно для кэширования слоя
COPY requirements.txt .

# Установка зависимостей в один слой для уменьшения размера
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY chunk_n_vec.py .
COPY config.py .
COPY secrets_app.py .

# Создание непривилегированного пользователя
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8998

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "chunk_n_vec:app", "--host", "0.0.0.0", "--port", "8998", "--log-level", "info"]