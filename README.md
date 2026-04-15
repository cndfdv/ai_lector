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
├── test_results.ipynb              # Тесты полного пайплайна на реальных аудио
├── mini_tests.ipynb                # Пошаговые тесты отдельных функций
└── test_rag.ipynb                  # Тесты RAG
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
POSTGRES_HOST=localhost

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
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

Принимает аудиофайл (mp3/wav), выполняет полный пайплайн: диаризация → транскрипция → LLM-анализ → генерация подкаста. Автоматически сохраняет лекцию в RAG.

Формат запроса: `multipart/form-data`.

| Поле | Тип | Описание |
|------|-----|----------|
| `file` | file, **required** | Аудиофайл лекции (`.mp3` или `.wav`) |
| `record_id` | string, **required** | Уникальный ID записи |
| `groups` | string[], **required** | Список студенческих групп (можно указать несколько раз) |
| `lecture_date` | string, optional | Дата лекции (`DD-MM-YYYY`), по умолчанию сегодня |

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@lecture.mp3" \
  -F "record_id=rec-001" \
  -F "groups=CS-101" \
  -F "groups=CS-102" \
  -F "lecture_date=15-01-2025"
```

Ответ (`AnalysisResponse`):

```jsonc
{
  // Полная транскрипция лекции (string)
  "lecture_text": "Добрый день, сегодня мы поговорим о...",

  // Конспект в формате markdown (string)
  "abstract_text": "# Тема лекции\n\n## Введение\n...",

  // Скорость речи по минутам: ключ — номер минуты, значение — слов/мин (object<string, float>)
  "speech_speed": {
    "0": 120.5,
    "1": 135.2,
    "2": 98.0
  },

  // Майнд-карта: иерархическая структура тем лекции (object)
  "mindmap": {
    "title": "Название лекции",
    "nodes": [
      {
        "id": "тема 1",
        "label": "Основная тема",
        "children": [
          {
            "id": "подтема 1.1",
            "label": "Подтема",
            "children": []
          }
        ]
      }
    ]
  },

  // Частотные слова БЕЗ стоп-слов: [аудитория, лектор] (array[object<string, int>])
  // Каждый элемент — словарь {слово: количество}, топ-10
  "popular_words_no_stopw": [
    {"нейронный": 12, "модель": 8},
    {"обучение": 25, "данные": 18}
  ],

  // Частотные слова СО стоп-словами: [аудитория, лектор] (array[object<string, int>])
  "popular_words_w_stopw": [
    {"и": 50, "в": 35, "что": 28},
    {"это": 45, "мы": 30, "и": 28}
  ],

  // Распределение времени лекции в процентах (object<string, float>)
  "conversation_static": {
    "lecturer": 75.0,   // % времени — лектор говорит
    "discussion": 15.0,  // % времени — диалог/вопросы аудитории
    "quiet": 10.0        // % времени — тишина/паузы
  },

  // Таймлайн лекции: список чанков (array[array])
  // Каждый элемент: [speaker_id, "текст фрагмента", "мин:сек"]
  // speaker_id: 1 = лектор, 2 = аудитория
  "lecture_timeline": [
    [1, "Добрый день, начнём лекцию", "0:0"],
    [2, "А можно вопрос?", "2:30"],
    [1, "Конечно, слушаю", "2:45"]
  ],

  // Вопросы для самопроверки, 12 штук (array[string])
  "questions": [
    "Что такое нейронная сеть?",
    "Какие функции активации существуют?"
  ],

  // Путь к сгенерированному подкасту, null если не удалось (string | null)
  "podcast": "a1b2c3d4.mp3"
}
```

#### `POST /analyze/async` — Асинхронный анализ (рекомендуется для фронта/мобайла)

Тот же пайплайн, но **не блокирует клиента** — сразу возвращает `task_id` (HTTP 202). Результат получать через `GET /tasks/{task_id}`.

Формат запроса: такой же как у `POST /analyze` (`multipart/form-data`).

```bash
curl -X POST http://localhost:8000/analyze/async \
  -F "file=@lecture.mp3" \
  -F "record_id=rec-001" \
  -F "groups=CS-101" \
  -F "groups=CS-102" \
  -F "lecture_date=15-01-2025"
```

Ответ (HTTP 202):

```json
{"task_id": "550e8400-e29b-41d4-a716-446655440000"}
```

#### `GET /tasks/{task_id}` — Статус задачи

Поллинг статуса асинхронного анализа. Рекомендуемый интервал — 5-10 секунд.

```bash
curl http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000
```

Ответ (`TaskInfo`):

```jsonc
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",

  // "pending" — в очереди, "processing" — обрабатывается,
  // "completed" — готово, "failed" — ошибка
  "status": "completed",

  // Результат анализа, появляется когда status = "completed"
  // Формат такой же как у POST /analyze (AnalysisResponse)
  "result": { ... },

  // Сообщение об ошибке, появляется когда status = "failed"
  "error": null
}
```

| status | result | error | Что делать |
|--------|--------|-------|------------|
| `pending` | null | null | Ждать, повторить запрос через 5-10 сек |
| `processing` | null | null | Ждать, повторить запрос через 5-10 сек |
| `completed` | `AnalysisResponse` | null | Готово, забрать результат из `result` |
| `failed` | null | string | Ошибка, показать `error` пользователю |

### RAG — Управление лекциями

#### `POST /lectures` — Добавить лекцию в RAG

```bash
curl -X POST http://localhost:8000/lectures \
  -H "Content-Type: application/json" \
  -d '{
    "lecture_text": "Текст лекции...",
    "student_groups": ["CS-101", "CS-102"],
    "lecture_date": "15-01-2025",
    "record_id": "rec-001"
  }'
```

Ответ:

```json
{"lecture_id": "550e8400-e29b-41d4-a716-446655440000"}
```

#### `GET /lectures` — Список лекций

| Параметр | Тип | Описание |
|----------|-----|----------|
| `student_group` | string, optional | Фильтр по студенческой группе |
| `limit` | int, default=100 (1–1000) | Максимум записей |

```bash
# Все лекции
curl http://localhost:8000/lectures

# По группе
curl "http://localhost:8000/lectures?student_group=CS-101&limit=10"
```

Ответ (`array[LectureInfo]`):

```json
[
  {
    "id": "550e8400-...",
    "record_id": "rec-001",
    "student_groups": ["CS-101", "CS-102"],
    "lecture_date": "15-01-2025"
  }
]
```

#### `GET /lectures/{id}` — Получить лекцию

```bash
curl http://localhost:8000/lectures/550e8400-...
```

Ответ (`LectureDetail`):

```json
{
  "id": "550e8400-...",
  "record_id": "rec-001",
  "student_groups": ["CS-101", "CS-102"],
  "lecture_date": "15-01-2025",
  "content": "Полный текст лекции..."
}
```

#### `DELETE /lectures/{id}` — Удалить лекцию

Удаляет лекцию и все её чанки из PostgreSQL и Milvus.

```bash
curl -X DELETE http://localhost:8000/lectures/550e8400-...
```

Ответ:

```json
{"deleted": true}
```

### RAG — Запросы

#### `POST /query` — Вопрос по лекциям (Agentic RAG)

Agentic RAG: переформулирует вопрос → ищет релевантные чанки в Milvus → генерирует ответ через LLM на основе найденного контекста.

| Поле | Тип | Описание |
|------|-----|----------|
| `question` | string, **required** | Вопрос по лекциям |
| `student_group` | string, optional | Фильтр — ответ только на основе лекций этой группы |

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Что такое нейронная сеть?", "student_group": "CS-101"}'
```

Ответ (`QueryResponse`):

```jsonc
{
  // Сгенерированный ответ на основе найденных фрагментов лекций (string)
  "answer": "Нейронная сеть — это математическая модель...",

  // Метаданные чанков, использованных для генерации ответа (array[object])
  "sources": [
    {
      "lecture_id": "550e8400-...",
      "student_groups": ["CS-101"],
      "lecture_date": "15-01-2025"
    }
  ],

  // Переформулированный запрос, использованный для поиска (string)
  "rewritten_question": "определение нейронной сети архитектура принцип работы"
}
```

#### `POST /search` — Семантический поиск

Поиск по similarity без генерации ответа. Возвращает топ-K релевантных чанков.

| Поле | Тип | Описание |
|------|-----|----------|
| `query` | string, **required** | Поисковый запрос |
| `student_group` | string, optional | Фильтр по студенческой группе |
| `k` | int, default=5 (1–50) | Количество результатов |

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "градиентный спуск", "student_group": "CS-101", "k": 5}'
```

Ответ (`array[SearchResult]`):

```json
[
  {
    "content": "Градиентный спуск — это итеративный алгоритм оптимизации...",
    "metadata": {
      "lecture_id": "550e8400-...",
      "student_groups": ["CS-101"],
      "lecture_date": "15-01-2025"
    }
  }
]
```

#### `POST /search/dates` — Поиск по диапазону дат

Семантический поиск только среди лекций за указанный период.

| Поле | Тип | Описание |
|------|-----|----------|
| `query` | string, **required** | Поисковый запрос |
| `start_date` | string, **required** | Начало периода (DD-MM-YYYY) |
| `end_date` | string, **required** | Конец периода (DD-MM-YYYY) |
| `student_group` | string, optional | Фильтр по студенческой группе |
| `k` | int, default=5 (1–50) | Количество результатов |

```bash
curl -X POST http://localhost:8000/search/dates \
  -H "Content-Type: application/json" \
  -d '{
    "query": "обратное распространение ошибки",
    "start_date": "01-01-2025",
    "end_date": "30-06-2025",
    "student_group": "CS-101",
    "k": 5
  }'
```

Ответ: такой же формат как у `POST /search` (`array[SearchResult]`).

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
| `milvus` | 19530 | Milvus (векторный поиск) |
| `etcd` | 2379 | etcd (для Milvus) |
| `minio` | 9001 | MinIO (хранилище для Milvus) |
| `attu` | 3000 | Attu (UI для управления Milvus) |

## Использование из Python

```python
from src.rec_analyzer import LectureAnalyzer

# Инициализация (RAG включён по умолчанию)
analyzer = LectureAnalyzer()

# Анализ лекции (автоматически сохраняется в RAG)
result = analyzer.process("lecture.mp3", "rec-001", ["CS-101"])

# Анализ без RAG
result = analyzer.process("lecture.mp3", "rec-002", ["CS-101"], use_rag=False)

# Запрос к RAG
answer = analyzer.rag.query("Что такое нейронная сеть?")
print(answer["answer"])
```
