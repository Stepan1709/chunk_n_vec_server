# Dockerfile
FROM python:3.11-slim

# Установка рабочей директории
WORKDIR /app

# Установка системных зависимостей, необходимых для numpy и chonkie
# build-essential = gcc/g++/make, libblas-dev/liblapack-dev нужны для numpy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip и установка wheel (чтобы по возможности ставились готовые колеса)
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Копирование файла с зависимостями
# Принудительное обновление chonkie до последней версии 1.6.1
RUN pip install --no-cache-dir --upgrade chonkie==1.6.1
COPY requirements.txt .

# Установка всех зависимостей
# Убираем --force-reinstall и --no-binary :all:, чтобы pip мог взять готовые колеса numpy
# Это решит проблему со сборкой и ускорит сборку
RUN pip install --no-cache-dir -r requirements.txt

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