# API Reference — AI Lector

Полная документация HTTP API сервиса AI Lector: транскрипция и анализ лекций, RAG-поиск по базе.

- [Общие соглашения](#общие-соглашения)
- [Анализ лекций](#анализ-лекций)
  - [`POST /analyze`](#post-analyze)
  - [`POST /analyze/async`](#post-analyzeasync)
  - [`GET /tasks/{task_id}`](#get-taskstask_id)
- [RAG — управление лекциями](#rag--управление-лекциями)
  - [`POST /lectures`](#post-lectures)
  - [`GET /lectures`](#get-lectures)
  - [`GET /lectures/{id}`](#get-lecturesid)
  - [`DELETE /lectures/{id}`](#delete-lecturesid)
- [RAG — запросы](#rag--запросы)
  - [`POST /query`](#post-query)
  - [`POST /search`](#post-search)
  - [`POST /search/dates`](#post-searchdates)
- [Pydantic-модели](#pydantic-модели)

---

## Общие соглашения

| Параметр | Значение |
|----------|----------|
| Base URL | `http://localhost:8000` |
| Content-Type ответов | `application/json` |
| Content-Type запросов | `application/json` (или `multipart/form-data` для `/analyze*`) |
| Кодировка | UTF-8 |

### Интерактивная документация

| URL | Описание |
|-----|----------|
| `/docs` | Swagger UI — интерактивный клиент |
| `/redoc` | ReDoc — читаемая reference |
| `/openapi.json` | OpenAPI 3.x спецификация (JSON) |

### Формат ошибок

Все ошибки возвращаются по стандарту FastAPI с HTTP-кодом 4xx/5xx и телом:

**Простые ошибки (400, 404, 500):**

```json
{"detail": "Текст ошибки"}
```

**Ошибки валидации (422 Unprocessable Entity):**

```json
{
  "detail": [
    {
      "loc": ["body", "lecture_date"],
      "msg": "Неверный формат даты: '2025-01-15', ожидается DD-MM-YYYY",
      "type": "value_error"
    }
  ]
}
```

### Общие коды ответов

| Код | Когда возвращается |
|-----|-------------------|
| `200` | Успешный ответ (по умолчанию) |
| `202` | Только `POST /analyze/async` — задача поставлена в очередь |
| `400` | Неверные данные запроса (формат файла, формат даты) |
| `404` | Сущность не найдена (лекция, задача) |
| `422` | Ошибка валидации Pydantic-схемы |
| `500` | Внутренняя ошибка: `{"detail": "Внутренняя ошибка сервера"}` |

---

## Анализ лекций

### `POST /analyze`

Полный пайплайн обработки аудиозаписи: диаризация → транскрипция → LLM-анализ → генерация подкаста. Автоматически сохраняет лекцию в RAG.

**Внимание:** обработка может занимать минуты. Для UI/мобильных клиентов используй [`POST /analyze/async`](#post-analyzeasync).

**Запрос:** `multipart/form-data`

| Поле | Тип | Обяз. | Описание |
|------|-----|:-----:|----------|
| `file` | file (`.mp3` / `.wav`) | да | Аудиофайл лекции |
| `record_id` | string | да | Уникальный ID записи |
| `groups` | string[] | да | Список студенческих групп (повторять поле для нескольких) |
| `lecture_date` | string | нет | Дата лекции `DD-MM-YYYY`, по умолчанию — сегодня |

**Пример запроса:**

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@lecture.mp3" \
  -F "record_id=rec-001" \
  -F "groups=CS-101" \
  -F "groups=CS-102" \
  -F "lecture_date=15-01-2025"
```

**Успешный ответ:** `200 OK` — `AnalysisResponse`

```jsonc
{
  // Полная транскрипция (string)
  "lecture_text": "Добрый день, сегодня мы поговорим о...",

  // Конспект лекции в формате markdown (string)
  "abstract_text": "# Тема лекции\n\n## Введение\n...",

  // Скорость речи: ключ — номер минуты, значение — слов/мин (object<string, float>)
  "speech_speed": {
    "0": 120.5,
    "1": 135.2,
    "2": 98.0
  },

  // Майнд-карта: иерархическая структура тем (object)
  "mindmap": {
    "title": "Название лекции",
    "nodes": [
      {
        "id": "тема 1",
        "label": "Основная тема",
        "children": [
          {"id": "подтема 1.1", "label": "Подтема", "children": []}
        ]
      }
    ]
  },

  // Частотные слова БЕЗ стоп-слов: [аудитория, лектор]
  // (array[object<string, int>], всегда ровно 2 элемента)
  "popular_words_no_stopw": [
    {"нейронный": 12, "модель": 8},
    {"обучение": 25, "данные": 18}
  ],

  // Частотные слова СО стоп-словами: [аудитория, лектор]
  "popular_words_w_stopw": [
    {"и": 50, "в": 35, "что": 28},
    {"это": 45, "мы": 30, "и": 28}
  ],

  // Распределение времени лекции в процентах (object<string, float>)
  "conversation_static": {
    "lecturer": 75.0,    // % — лектор говорит
    "discussion": 15.0,  // % — диалог / вопросы аудитории
    "quiet": 10.0        // % — тишина
  },

  // Таймлайн лекции (array[array])
  // Каждый элемент: [speaker_id (int), "текст" (string), "мин:сек" (string)]
  // speaker_id: 1 = лектор, 2 = аудитория
  "lecture_timeline": [
    [1, "Добрый день, начнём лекцию", "0:0"],
    [2, "А можно вопрос?", "2:30"],
    [1, "Конечно, слушаю", "2:45"]
  ],

  // Вопросы для самопроверки, 10–12 шт. (array[string])
  "questions": [
    "Что такое нейронная сеть?",
    "Какие функции активации существуют?"
  ],

  // Имя файла подкаста (mp3) или null если генерация не удалась (string | null)
  "podcast": "a1b2c3d4.mp3"
}
```

**Коды ошибок:**

| Код | Условие | Тело |
|-----|---------|------|
| `400` | Файл не `.mp3` / `.wav` | `{"detail": "Поддерживаются только mp3 и wav файлы"}` |
| `400` | Неверный формат даты | `{"detail": "Неверный формат даты, ожидается DD-MM-YYYY"}` |
| `404` | Аудиофайл не найден внутри сервиса | `{"detail": "Аудиофайл не найден"}` |
| `500` | Внутренняя ошибка анализа | `{"detail": "Внутренняя ошибка сервера"}` |

---

### `POST /analyze/async`

Тот же пайплайн, что и [`POST /analyze`](#post-analyze), но **не блокирует клиента** — сразу возвращает `task_id`. Результат получать через [`GET /tasks/{task_id}`](#get-taskstask_id). Рекомендованный способ для UI и мобильных клиентов.

**Запрос:** идентичен [`POST /analyze`](#post-analyze) (`multipart/form-data` с теми же полями).

**Пример запроса:**

```bash
curl -X POST http://localhost:8000/analyze/async \
  -F "file=@lecture.mp3" \
  -F "record_id=rec-001" \
  -F "groups=CS-101" \
  -F "lecture_date=15-01-2025"
```

**Успешный ответ:** `202 Accepted` — `TaskSubmitResponse`

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Коды ошибок:** те же, что и у [`POST /analyze`](#post-analyze) (400 для валидации входа). Ошибки самого анализа не возвращаются здесь — они доставляются через `GET /tasks/{task_id}` со статусом `failed`.

---

### `GET /tasks/{task_id}`

Поллинг статуса асинхронного анализа. Рекомендуемый интервал — 5–10 секунд. Задача хранится в памяти 1 час с момента создания.

**Параметр пути:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `task_id` | string (UUID) | ID, выданный `POST /analyze/async` |

**Пример запроса:**

```bash
curl http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000
```

**Успешный ответ:** `200 OK` — `TaskInfo`

```jsonc
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",

  // pending | processing | completed | failed
  "status": "completed",

  // AnalysisResponse — заполнено только при status="completed"
  "result": { /* см. POST /analyze */ },

  // Сообщение об ошибке — только при status="failed"
  "error": null
}
```

**Что хранится в полях по статусам:**

| `status` | `result` | `error` | Действие клиента |
|----------|----------|---------|------------------|
| `pending` | `null` | `null` | Ждать, повторить запрос через 5–10 сек |
| `processing` | `null` | `null` | Ждать, повторить запрос через 5–10 сек |
| `completed` | `AnalysisResponse` | `null` | Забрать результат из `result` |
| `failed` | `null` | string | Показать `error` пользователю |

**Коды ошибок:**

| Код | Условие | Тело |
|-----|---------|------|
| `404` | Задача не найдена или истёк TTL (1 ч) | `{"detail": "Задача не найдена"}` |

---

## RAG — управление лекциями

### `POST /lectures`

Чанкует текст лекции, считает эмбеддинги и сохраняет в PostgreSQL + Milvus для последующего поиска и Q&A.

**Запрос:** `application/json` — `AddLectureRequest`

| Поле | Тип | Обяз. | Описание |
|------|-----|:-----:|----------|
| `lecture_text` | string | да | Полный текст лекции |
| `student_groups` | string[] | да | Список студенческих групп |
| `lecture_date` | string | да | Дата лекции `DD-MM-YYYY` |
| `record_id` | string \| null | нет | Уникальный ID записи |

**Пример запроса:**

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

**Успешный ответ:** `200 OK` — `AddLectureResponse`

```json
{
  "lecture_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Коды ошибок:**

| Код | Условие |
|-----|---------|
| `422` | Невалидный формат `lecture_date` или другие поля |
| `500` | Ошибка чанкинга / БД |

---

### `GET /lectures`

Возвращает список лекций (без полного текста) с опциональной фильтрацией по группе.

**Query-параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `student_group` | string | — | Фильтр по группе |
| `limit` | int (1–1000) | 100 | Максимум записей |

**Пример запроса:**

```bash
# Все лекции
curl http://localhost:8000/lectures

# По группе с лимитом
curl "http://localhost:8000/lectures?student_group=CS-101&limit=10"
```

**Успешный ответ:** `200 OK` — `List[LectureInfo]`

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "record_id": "rec-001",
    "student_groups": ["CS-101", "CS-102"],
    "lecture_date": "15-01-2025"
  }
]
```

**Коды ошибок:**

| Код | Условие |
|-----|---------|
| `422` | `limit` вне диапазона 1–1000 |

---

### `GET /lectures/{id}`

Возвращает полную информацию о лекции, включая весь её текст.

**Параметр пути:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `id` | string (UUID) | ID лекции |

**Пример запроса:**

```bash
curl http://localhost:8000/lectures/550e8400-e29b-41d4-a716-446655440000
```

**Успешный ответ:** `200 OK` — `LectureDetail`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "record_id": "rec-001",
  "student_groups": ["CS-101", "CS-102"],
  "lecture_date": "15-01-2025",
  "content": "Полный текст лекции..."
}
```

**Коды ошибок:**

| Код | Условие | Тело |
|-----|---------|------|
| `404` | Лекция не найдена | `{"detail": "Лекция не найдена"}` |

---

### `DELETE /lectures/{id}`

Удаляет лекцию и все её чанки из PostgreSQL и Milvus. Операция необратима.

**Параметр пути:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `id` | string (UUID) | ID лекции |

**Пример запроса:**

```bash
curl -X DELETE http://localhost:8000/lectures/550e8400-e29b-41d4-a716-446655440000
```

**Успешный ответ:** `200 OK` — `DeleteResponse`

```json
{"deleted": true}
```

**Коды ошибок:**

| Код | Условие | Тело |
|-----|---------|------|
| `404` | Лекция не найдена | `{"detail": "Лекция не найдена"}` |

---

## RAG — запросы

### `POST /query`

Agentic RAG: переформулирует вопрос → ищет релевантные чанки в Milvus → генерирует ответ через LLM на основе найденного контекста.

**Запрос:** `application/json` — `QueryRequest`

| Поле | Тип | Обяз. | Описание |
|------|-----|:-----:|----------|
| `question` | string | да | Вопрос по лекциям |
| `student_group` | string \| null | нет | Фильтр — отвечать только по лекциям этой группы |

**Пример запроса:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Что такое нейронная сеть?", "student_group": "CS-101"}'
```

**Успешный ответ:** `200 OK` — `QueryResponse`

```jsonc
{
  // Сгенерированный ответ (string)
  "answer": "Нейронная сеть — это математическая модель...",

  // Метаданные использованных чанков-источников (array[object])
  "sources": [
    {
      "lecture_id": "550e8400-e29b-41d4-a716-446655440000",
      "student_groups": ["CS-101"],
      "lecture_date": "15-01-2025"
    }
  ],

  // Запрос после переформулировки агентом — то, что реально пошло в поиск (string)
  "rewritten_question": "определение нейронной сети архитектура принцип работы"
}
```

**Структура одного элемента `sources`:**

| Поле | Тип | Описание |
|------|-----|----------|
| `lecture_id` | string (UUID) | ID лекции-источника |
| `student_groups` | string[] | Группы лекции |
| `lecture_date` | string | Дата `DD-MM-YYYY` |

**Коды ошибок:**

| Код | Условие |
|-----|---------|
| `422` | Поле `question` пустое / отсутствует |
| `500` | Ошибка LLM или векторного поиска |

---

### `POST /search`

Семантический поиск по лекциям без генерации ответа. Возвращает топ-K чанков по сходству эмбеддинга. Опционально фильтрует по группе.

**Запрос:** `application/json` — `SearchRequest`

| Поле | Тип | Обяз. | По умолч. | Описание |
|------|-----|:-----:|-----------|----------|
| `query` | string | да | — | Поисковый запрос |
| `student_group` | string \| null | нет | — | Фильтр по группе |
| `k` | int (1–50) | нет | 5 | Кол-во результатов |

**Пример запроса:**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "градиентный спуск", "student_group": "CS-101", "k": 5}'
```

**Успешный ответ:** `200 OK` — `List[SearchResult]`

```json
[
  {
    "content": "Градиентный спуск — это итеративный алгоритм оптимизации...",
    "metadata": {
      "lecture_id": "550e8400-e29b-41d4-a716-446655440000",
      "student_groups": ["CS-101"],
      "lecture_date": "15-01-2025"
    }
  }
]
```

**Структура `metadata`:**

| Поле | Тип | Описание |
|------|-----|----------|
| `lecture_id` | string (UUID) | ID лекции-источника |
| `student_groups` | string[] | Группы лекции |
| `lecture_date` | string | Дата `DD-MM-YYYY` |

**Коды ошибок:**

| Код | Условие |
|-----|---------|
| `422` | `k` вне диапазона 1–50, пустой `query` |

---

### `POST /search/dates`

Семантический поиск среди лекций за указанный период. По формату идентичен [`POST /search`](#post-search), но добавляет фильтр по диапазону дат.

**Запрос:** `application/json` — `DateSearchRequest`

| Поле | Тип | Обяз. | По умолч. | Описание |
|------|-----|:-----:|-----------|----------|
| `query` | string | да | — | Поисковый запрос |
| `start_date` | string | да | — | Начало периода `DD-MM-YYYY` |
| `end_date` | string | да | — | Конец периода `DD-MM-YYYY` |
| `student_group` | string \| null | нет | — | Фильтр по группе |
| `k` | int (1–50) | нет | 5 | Кол-во результатов |

**Пример запроса:**

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

**Успешный ответ:** `200 OK` — `List[SearchResult]`

Формат ответа полностью идентичен [`POST /search`](#post-search):

```json
[
  {
    "content": "Обратное распространение ошибки (backpropagation) — это алгоритм...",
    "metadata": {
      "lecture_id": "550e8400-e29b-41d4-a716-446655440000",
      "student_groups": ["CS-101"],
      "lecture_date": "20-03-2025"
    }
  }
]
```

**Коды ошибок:**

| Код | Условие | Тело |
|-----|---------|------|
| `422` | Неверный формат `start_date` / `end_date` | `{"detail": [...]}` |
| `422` | `k` вне диапазона 1–50 | `{"detail": [...]}` |

---

## Pydantic-модели

Все модели определены в [app.py](../app.py). Ниже — компактная reference со ссылками на исходник.

### Запросы

| Модель | Файл / строки | Эндпоинты |
|--------|---------------|-----------|
| `AddLectureRequest` | [app.py:26-43](../app.py#L26-L43) | `POST /lectures` |
| `QueryRequest` | [app.py:46-50](../app.py#L46-L50) | `POST /query` |
| `SearchRequest` | [app.py:53-58](../app.py#L53-L58) | `POST /search` |
| `DateSearchRequest` | [app.py:61-79](../app.py#L61-L79) | `POST /search/dates` |

### Ответы

| Модель | Файл / строки | Эндпоинты |
|--------|---------------|-----------|
| `AnalysisResponse` | [app.py:87-112](../app.py#L87-L112) | `POST /analyze`, в `TaskInfo.result` |
| `LectureInfo` | [app.py:115-121](../app.py#L115-L121) | `GET /lectures` |
| `LectureDetail` | [app.py:124-131](../app.py#L124-L131) | `GET /lectures/{id}` |
| `QueryResponse` | [app.py:134-143](../app.py#L134-L143) | `POST /query` |
| `SearchResult` | [app.py:146-150](../app.py#L146-L150) | `POST /search`, `POST /search/dates` |
| `AddLectureResponse` | [app.py:153-156](../app.py#L153-L156) | `POST /lectures` |
| `DeleteResponse` | [app.py:159-162](../app.py#L159-L162) | `DELETE /lectures/{id}` |
| `TaskSubmitResponse` | [app.py:177-180](../app.py#L177-L180) | `POST /analyze/async` |
| `TaskInfo` | [app.py:183-189](../app.py#L183-L189) | `GET /tasks/{task_id}` |

### Энумы

| Енум | Файл / строки | Значения |
|------|---------------|----------|
| `TaskStatus` | [app.py:170-174](../app.py#L170-L174) | `pending` · `processing` · `completed` · `failed` |
