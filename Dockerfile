# Dockerfile
FROM python:3.11-slim

# Установка рабочей директории
WORKDIR /app

# Установка минимальных системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование файла с зависимостями
COPY requirements.txt .

# Установка всех зависимостей (numpy 2.x установится автоматически)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY chunk_n_vec.py .
COPY config.py .
COPY secrets.py .

# Создание пользователя без прав root для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Открытие порта
EXPOSE 8998

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Запуск приложения
CMD ["python", "-m", "uvicorn", "chunk_n_vec:app", "--host", "0.0.0.0", "--port", "8998", "--log-level", "info"]