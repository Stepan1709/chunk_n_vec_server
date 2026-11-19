# chunk_n_vec_server

Микросервис для разбиения текста на чанки и их векторизации. Часть RAG-пайплайна:

```
S3 (MinIO) ──> s3_to_qdrant_synchronizer ──> file_to_text_converter_server ──> PaddleOCR-VL_pdf_ocr_server
                         │
                         └──> chunk_n_vec_server (этот сервис) ──> сервис эмбеддингов (vLLM, OpenAI API)
                         │
                         └──> Qdrant
```

Сервис принимает текст, режет его на чанки фиксированного размера с перекрытием и для каждого чанка получает
эмбеддинг через OpenAI-совместимый эндпоинт `POST {VEC_URL}/v1/embeddings` (например, vLLM с моделью `BAAI/bge-m3`).

## Алгоритм чанкинга

- Размер чанка — `max_chunk_size` символов плюс `overlap` символов перекрытия: чанк `i` содержит символы
  `[i*max_chunk_size : (i+1)*max_chunk_size + overlap]`.
- Текст короче `max_chunk_size + overlap` возвращается одним чанком.
- Последний хвост, который целиком входит в перекрытие предыдущего чанка (не длиннее `overlap`), не сохраняется.
- Должно выполняться `overlap < max_chunk_size`, иначе возвращается `400`.

## Конфигурация

Все настройки задаются переменными окружения (шаблон — `.env.example`). Файл `.env` в git не попадает.

| Переменная                          | По умолчанию            | Описание                                              |
|-------------------------------------|-------------------------|-------------------------------------------------------|
| `VEC_URL`                           | `http://localhost:8500` | URL сервиса эмбеддингов                               |
| `EMBEDDING_MODEL`                   | `BAAI/bge-m3`           | Модель эмбеддингов                                    |
| `REQUEST_TIMEOUT`                   | `120`                   | Таймаут запроса к сервису эмбеддингов, сек            |
| `MAX_CONCURRENT_EMBEDDING_REQUESTS` | `5`                     | Максимум параллельных запросов к сервису эмбеддингов  |
| `SERVER_HOST`                       | `0.0.0.0`               | Адрес, на котором слушает сервис                      |
| `SERVER_PORT`                       | `8998`                  | Порт сервиса (в Docker-образе проброшен `8998`)       |
| `LOG_LEVEL`                         | `INFO`                  | Уровень логирования                                   |

## Запуск

### Docker Compose

```bash
cp .env.example .env     # и отредактируйте значения
docker compose up -d --build
docker compose logs -f
```

### Docker

```bash
docker build -t chunk-n-vec-server .
docker run -d --name chunk-n-vec-server -p 8998:8998 --env-file .env --restart unless-stopped chunk-n-vec-server
```

### Локально

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export VEC_URL=http://<host>:8500                   # Windows PowerShell: $env:VEC_URL="..."
python -m app
```

## API

| Метод  | Путь       | Описание                                                                         |
|--------|------------|----------------------------------------------------------------------------------|
| `GET`  | `/`        | Информация о сервисе                                                             |
| `GET`  | `/live`    | Liveness-проба (используется в `HEALTHCHECK`, внешние сервисы не проверяет)      |
| `GET`  | `/health`  | Состояние сервиса и доступность сервиса эмбеддингов (`healthy` / `degraded`)     |
| `POST` | `/process` | Чанкинг и векторизация текста                                                    |

### `POST /process`

Тело запроса (JSON):

| Поле             | Тип | По умолчанию | Описание                                        |
|------------------|-----|--------------|-------------------------------------------------|
| `text`           | str | —            | Текст (не пустой)                               |
| `max_chunk_size` | int | `4000`       | Размер чанка без перекрытия, 100–10000 символов |
| `overlap`        | int | `500`        | Перекрытие, 0–2000 символов                     |

```bash
curl -X POST http://localhost:8998/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Длинный текст...", "max_chunk_size": 4000, "overlap": 500}'
```

Ответ:

```json
{
  "chunks": [
    {"embedding": [0.012, -0.034, "..."], "chunk_text": "Первый чанк..."}
  ],
  "total_chunks": 1,
  "processing_time": 0.412
}
```

Коды ошибок: `400` — некорректные параметры чанкинга, `503` / `504` — сервис эмбеддингов недоступен / не ответил
вовремя, `502` — сервис эмбеддингов вернул ошибку или некорректный ответ, `500` — прочие ошибки.

## Структура проекта

```
app/
├── main.py            # сборка приложения и lifespan (создание HTTP-клиента)
├── __main__.py        # запуск: python -m app
├── core/              # конфигурация, логирование, исключения
├── api/
│   ├── deps.py        # зависимости FastAPI
│   └── routers/       # HTTP-эндпоинты (health, process)
├── schemas/           # pydantic-схемы запросов и ответов
├── services/          # бизнес-логика: чанкинг и векторизация
└── clients/           # клиент сервиса эмбеддингов
tests/                 # pytest
```

## Разработка

```bash
pip install -r requirements-dev.txt
pre-commit install          # ruff и mypy перед каждым коммитом
ruff check . && ruff format --check . && mypy app
pytest
```

Те же проверки запускаются в GitHub Actions (`.github/workflows/ci.yml`).
