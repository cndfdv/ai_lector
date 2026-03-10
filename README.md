# AI Lector

Сервис анализа лекций: транскрипция аудиозаписей, генерация конспектов, майнд-карт, вопросов для самопроверки, подкастов и поиск по базе лекций (RAG).

## Структура проекта

```
ai_lector/
├── app.py                          # FastAPI приложение (точка входа)
├── Dockerfile                      # Образ сервиса (PyTorch + CUDA)
├── docker-compose.yml              # PostgreSQL, Milvus, App
├── requirements.txt                # Зависимости Python
├── .env                            # Конфигурация (порты, ключи, модели)
│
├── src/
│   ├── rec_analyzer.py             # LectureAnalyzer — основной класс анализа
│   ├── llm_models.py               # Pydantic модели (Question, PodcastScript)
│   ├── prompts.py                  # Промпты для LLM
│   ├── stopwords.txt               # Стоп-слова (русский)
│   │
│   └── rag/                        # RAG модуль
│       ├── __init__.py             # LectureRAG — основной класс RAG
│       ├── agentic_rag.py          # LangGraph: rewrite → retrieve → generate
│       ├── retriever.py            # Поиск: similarity, по группе, по датам
│       ├── config.py               # RAGConfig из .env
│       ├── models.py               # SQLAlchemy: Lecture, LectureChunk
│       └── prompts.py              # Промпты для RAG
│
├── F5TTS/                          # Text-to-Speech (голосовой клонинг)
│   ├── f5_tts/                     # Модуль F5TTS
│   │   ├── api.py                  # F5TTS API
│   │   ├── model/                  # Архитектура модели (DiT, CFM)
│   │   └── infer/                  # Инференс
│   └── ckpts/                      # Чекпоинты модели (~1.3 GB)
│
├── utils/
│   └── podcast_host.wav            # Референсный голос ведущего подкаста
│
├── init/
│   └── 01_init.sql                 # Инициализация БД (таблицы, индексы)
│
├── data/txt_data/                  # Текстовые данные лекций
├── test.ipynb                      # Тесты на реальных аудио
└── build_rag.ipynb                 # Тесты RAG
```

## Архитектура

```
                     POST /analyze
                          │
                          ▼
                   ┌──────────────┐
                   │ LectureAnalyzer │
                   └──────┬───────┘
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    [Диаризация]    [Транскрипция]   [LLM анализ]
     pyannote         Whisper        ├── Конспект
                                     ├── Вопросы
                                     ├── Майнд-карта
                                     └── Подкаст скрипт
                                              │
                                              ▼
                                        [F5TTS + RUAccent]
                                         Генерация подкаста

                     POST /query
                          │
                          ▼
                   ┌──────────────┐
                   │  LectureRAG  │
                   └──────┬───────┘
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    [Rewrite Query] [Retrieve Top-K]  [Generate]
      LangGraph       Milvus +         LLM
                     PostgreSQL
```

## Быстрый старт

### 1. Настройка окружения

Скопировать `.env` и заполнить:

```env
# Database
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=lectures
POSTGRES_PORT=5433
POSTGRES_HOST=pg

# Milvus
MILVUS_HOST=milvus
MILVUS_PORT=19531
MILVUS_COLLECTION=lectures

# LLM
LLM_URL=http://10.162.1.92:1234/v1
LLM_NAME=gpt-oss-lab
LLM_API_KEY=not-needed

# HuggingFace (для pyannote)
HUGGINGFACEHUB_API_TOKEN=hf_...

# Embeddings
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```

### 2. Запуск через Docker

```bash
docker-compose up --build
```

Сервис будет доступен на `http://localhost:8000`.

Swagger UI (документация): `http://localhost:8000/docs`

### 3. Запуск без Docker

```bash
# Поднять БД
docker-compose up -d pg milvus etcd minio

# Установить зависимости
pip install -r requirements.txt

# Запустить сервис
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

### Анализ лекций

#### `POST /analyze` — Полный анализ аудиозаписи

Принимает аудиофайл (mp3/wav), выполняет полный пайплайн анализа.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@lecture.mp3" \
  -F "record_id=rec-001" \
  -F "groups=CS-101" \
  -F "groups=CS-102" \
  -F "lecture_date=2025-01-15"
```

Ответ:

```json
{
  "lecture_text": "Полная транскрипция лекции...",
  "abstract_text": "# Конспект\n...",
  "speech_speed": {"0": 120.5, "1": 135.2},
  "mindmap": {"title": "...", "nodes": [...]},
  "popular_words_no_stopw": [{}, {"слово": 10}],
  "popular_words_w_stopw": [{}, {"и": 50}],
  "conversation_static": {"lecturer": 75.0, "discussion": 15.0, "quiet": 10.0},
  "lecture_timeline": [[1, "текст чанка", "2:30"], ...],
  "questions": ["Что такое...?", "Какие методы...?"],
  "podcast": "uuid.mp3"
}
```

### RAG — Управление лекциями

#### `POST /lectures` — Добавить лекцию в RAG

```bash
curl -X POST http://localhost:8000/lectures \
  -H "Content-Type: application/json" \
  -d '{
    "lecture_text": "Текст лекции...",
    "student_groups": ["CS-101", "CS-102"],
    "lecture_date": "2025-01-15",
    "record_id": "rec-001"
  }'
```

#### `GET /lectures` — Список лекций

```bash
# Все лекции
curl http://localhost:8000/lectures

# По группе
curl "http://localhost:8000/lectures?student_group=CS-101&limit=10"
```

#### `GET /lectures/{id}` — Получить лекцию

```bash
curl http://localhost:8000/lectures/uuid-here
```

#### `DELETE /lectures/{id}` — Удалить лекцию

```bash
curl -X DELETE http://localhost:8000/lectures/uuid-here
```

### RAG — Запросы

#### `POST /query` — Вопрос по лекциям

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Что такое нейронная сеть?"}'
```

Ответ:

```json
{
  "answer": "Нейронная сеть — это...",
  "sources": [
    {"lecture_id": "...", "student_group": "CS-101", "lecture_date": "2025-01-15"}
  ],
  "rewritten_question": "определение нейронной сети архитектура"
}
```

#### `POST /search` — Similarity search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "градиентный спуск", "k": 5}'
```

#### `POST /search/group` — Поиск по группе

```bash
curl -X POST http://localhost:8000/search/group \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "student_group": "CS-101", "k": 5}'
```

#### `POST /search/dates` — Поиск по датам

```bash
curl -X POST http://localhost:8000/search/dates \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "start_date": "2025-01-01", "end_date": "2025-06-30", "k": 5}'
```

## Стек

| Компонент | Технология |
|-----------|-----------|
| API | FastAPI |
| Транскрипция | Whisper large-v3 |
| Диаризация | pyannote 3.1 |
| LLM | OpenAI-compatible (reasoning mode) |
| TTS | F5TTS + RUAccent |
| Embeddings | Qwen3-Embedding-0.6B |
| Chunking | chonkie (SlumberChunker + OpenAIGenie) |
| Оркестрация RAG | LangGraph |
| Векторная БД | Milvus |
| Реляционная БД | PostgreSQL |
| ORM | SQLAlchemy |

## Инфраструктура (docker-compose)

| Сервис | Порт (внешний) | Назначение |
|--------|---------------|-----------|
| `app` | 8000 | FastAPI сервис |
| `pg` | 5433 | PostgreSQL (метаданные лекций) |
| `milvus` | 19531 | Milvus (векторный поиск) |
| `etcd` | 2379 | etcd (для Milvus) |
| `minio` | 9001 | MinIO (хранилище для Milvus) |

## Использование из Python

```python
from src.rec_analyzer import LectureAnalyzer
from src.rag import LectureRAG
from datetime import date

# Инициализация
analyzer = LectureAnalyzer()
rag = LectureRAG()

# Анализ лекции
result = analyzer.process("lecture.mp3", "rec-001", ["CS-101"])

# Добавление в RAG
lecture_id = rag.add_lecture(
    lecture_text=result["lecture_text"],
    student_groups=["CS-101"],
    lecture_date=date.today(),
    record_id="rec-001",
)

# Запрос к RAG
answer = rag.query("Что такое нейронная сеть?")
print(answer["answer"])
```
