"""FastAPI application for AI Lector — lecture analysis and RAG service."""

import asyncio
import datetime
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.rec_analyzer import LectureAnalyzer

logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models — Request
# =============================================================================


class AddLectureRequest(BaseModel):
    """Запрос на добавление лекции в RAG."""

    lecture_text: str = Field(..., description="Полный текст лекции")
    student_groups: List[str] = Field(..., description="Список студенческих групп")
    lecture_date: str = Field(..., description="Дата лекции (DD-MM-YYYY)")
    record_id: Optional[str] = Field(None, description="Уникальный ID записи")

    @field_validator("lecture_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if len(v) != 10:
            raise ValueError(f"Неверный формат даты: {v!r}, ожидается DD-MM-YYYY")
        try:
            datetime.date(int(v[6:10]), int(v[3:5]), int(v[0:2]))
        except (ValueError, IndexError):
            raise ValueError(f"Неверный формат даты: {v!r}, ожидается DD-MM-YYYY")
        return v


class QueryRequest(BaseModel):
    """Запрос к RAG системе."""

    question: str = Field(..., description="Вопрос по лекциям")
    student_group: Optional[str] = Field(None, description="Фильтр по студенческой группе")


class SearchRequest(BaseModel):
    """Запрос на поиск по similarity."""

    query: str = Field(..., description="Поисковый запрос")
    student_group: Optional[str] = Field(None, description="Фильтр по студенческой группе")
    k: int = Field(5, ge=1, le=50, description="Количество результатов")


class DateSearchRequest(BaseModel):
    """Запрос на поиск по диапазону дат."""

    query: str = Field(..., description="Поисковый запрос")
    start_date: str = Field(..., description="Начальная дата (DD-MM-YYYY)")
    end_date: str = Field(..., description="Конечная дата (DD-MM-YYYY)")
    student_group: Optional[str] = Field(None, description="Фильтр по студенческой группе")
    k: int = Field(5, ge=1, le=50, description="Количество результатов")

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if len(v) != 10:
            raise ValueError(f"Неверный формат даты: {v!r}, ожидается DD-MM-YYYY")
        try:
            datetime.date(int(v[6:10]), int(v[3:5]), int(v[0:2]))
        except (ValueError, IndexError):
            raise ValueError(f"Неверный формат даты: {v!r}, ожидается DD-MM-YYYY")
        return v


# =============================================================================
# Pydantic Models — Response
# =============================================================================


class AnalysisResponse(BaseModel):
    """Результат полного анализа лекции."""

    lecture_text: str = Field(..., description="Транскрипция лекции")
    abstract_text: str = Field(..., description="Конспект лекции (markdown)")
    speech_speed: Dict[str, float] = Field(
        ..., description="Скорость речи по минутам (сл/мин)"
    )
    mindmap: Dict[str, Any] = Field(..., description="Структура майнд-карты")
    popular_words_no_stopw: List[Dict[str, int]] = Field(
        ..., description="Частые слова без стоп-слов [аудитория, лектор]"
    )
    popular_words_w_stopw: List[Dict[str, int]] = Field(
        ..., description="Частые слова со стоп-словами [аудитория, лектор]"
    )
    conversation_static: Dict[str, float] = Field(
        ...,
        description="Распределение времени: lecturer %, discussion %, quiet %",
    )
    lecture_timeline: List[List] = Field(
        ..., description="Таймлайн лекции (чанки с таймкодами)"
    )
    questions: List[str] = Field(
        ..., description="Вопросы для самопроверки (10-12 шт)"
    )
    podcast: Optional[str] = Field(None, description="Путь к сгенерированному подкасту (mp3), null если не удалось")


class LectureInfo(BaseModel):
    """Краткая информация о лекции."""

    id: str = Field(..., description="UUID лекции")
    record_id: Optional[str] = Field(None, description="ID записи")
    student_groups: List[str] = Field(..., description="Студенческие группы")
    lecture_date: str = Field(..., description="Дата лекции (DD-MM-YYYY)")


class LectureDetail(BaseModel):
    """Полная информация о лекции."""

    id: str
    record_id: Optional[str] = None
    student_groups: List[str]
    lecture_date: str = Field(..., description="Дата лекции (DD-MM-YYYY)")
    content: str = Field(..., description="Полный текст лекции")


class QueryResponse(BaseModel):
    """Ответ RAG системы."""

    answer: str = Field(..., description="Сгенерированный ответ")
    sources: List[Dict[str, Any]] = Field(
        ..., description="Метаданные использованных источников"
    )
    rewritten_question: str = Field(
        ..., description="Переформулированный запрос для поиска"
    )


class SearchResult(BaseModel):
    """Результат поиска."""

    content: str = Field(..., description="Текст чанка")
    metadata: Dict[str, Any] = Field(..., description="Метаданные чанка")


class AddLectureResponse(BaseModel):
    """Ответ на добавление лекции."""

    lecture_id: str = Field(..., description="UUID добавленной лекции")


class DeleteResponse(BaseModel):
    """Ответ на удаление."""

    deleted: bool


# =============================================================================
# Async task models
# =============================================================================


class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TaskSubmitResponse(BaseModel):
    """Ответ при постановке задачи в очередь."""

    task_id: str = Field(..., description="UUID задачи для отслеживания")


class TaskInfo(BaseModel):
    """Статус и результат задачи анализа."""

    task_id: str = Field(..., description="UUID задачи")
    status: TaskStatus = Field(..., description="Статус: pending, processing, completed, failed")
    result: Optional[AnalysisResponse] = Field(None, description="Результат (когда status=completed)")
    error: Optional[str] = Field(None, description="Сообщение об ошибке (когда status=failed)")


# In-memory task storage
_tasks: Dict[str, dict] = {}
_TASK_TTL_SECONDS = 3600  # 1 час


async def _cleanup_tasks():
    """Periodically remove tasks older than TTL."""
    while True:
        await asyncio.sleep(300)  # проверка каждые 5 минут
        now = datetime.datetime.utcnow()
        expired = [
            tid for tid, t in _tasks.items()
            if (now - t["created_at"]).total_seconds() > _TASK_TTL_SECONDS
        ]
        for tid in expired:
            del _tasks[tid]
        if expired:
            logger.info("Очищено %d устаревших задач", len(expired))


# =============================================================================
# Application
# =============================================================================

UPLOAD_DIR = "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка ML моделей при старте, очистка при остановке."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.state.analyzer = LectureAnalyzer()
    cleanup = asyncio.create_task(_cleanup_tasks())
    yield
    cleanup.cancel()


app = FastAPI(
    title="AI Lector",
    description=(
        "Сервис анализа лекций: транскрипция, конспект, майнд-карта, "
        "вопросы для самопроверки, подкаст, RAG по базе лекций."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — log details, return generic message."""
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("Необработанная ошибка: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера"})


# =============================================================================
# Endpoints — Анализ
# =============================================================================


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Анализ аудиозаписи лекции",
    description=(
        "Полный пайплайн: диаризация → транскрипция → LLM-анализ → подкаст. "
        "Принимает аудиофайл (mp3/wav), возвращает все результаты анализа."
    ),
    tags=["Анализ"],
)
async def analyze_lecture(
    file: UploadFile = File(..., description="Аудиофайл лекции (mp3 или wav)"),
    record_id: str = Form(..., description="Уникальный ID записи"),
    groups: List[str] = Form(..., description="Список студенческих групп"),
    lecture_date: Optional[str] = Form(
        None, description="Дата лекции (DD-MM-YYYY), по умолчанию сегодня"
    ),
):
    logger.info("POST /analyze record_id=%s groups=%s", record_id, groups)
    ext = os.path.splitext(file.filename or "audio.mp3")[1].lower()
    if ext not in (".mp3", ".wav"):
        raise HTTPException(400, "Поддерживаются только mp3 и wav файлы")

    with tempfile.NamedTemporaryFile(
        dir=UPLOAD_DIR, suffix=ext, delete=False
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    if lecture_date:
        try:
            date = datetime.date(
                int(lecture_date[6:10]), int(lecture_date[3:5]), int(lecture_date[0:2])
            )
        except (ValueError, IndexError):
            raise HTTPException(400, "Неверный формат даты, ожидается DD-MM-YYYY")
    else:
        date = datetime.date.today()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, app.state.analyzer.process, tmp_path, record_id, groups, date
        )
    except FileNotFoundError:
        raise HTTPException(404, "Аудиофайл не найден")
    except Exception:
        logger.exception("Ошибка анализа записи %s", record_id)
        raise HTTPException(500, "Внутренняя ошибка сервера")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return AnalysisResponse(**result)


# =============================================================================
# Endpoints — Асинхронный анализ
# =============================================================================


async def _run_analysis(task_id: str, tmp_path: str, record_id: str, groups: list, date):
    """Background worker: run analysis and update task store."""
    _tasks[task_id]["status"] = TaskStatus.processing
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, app.state.analyzer.process, tmp_path, record_id, groups, date
        )
        _tasks[task_id]["status"] = TaskStatus.completed
        _tasks[task_id]["result"] = result
    except Exception:
        logger.exception("Async-анализ %s (task %s) завершился ошибкой", record_id, task_id)
        _tasks[task_id]["status"] = TaskStatus.failed
        _tasks[task_id]["error"] = "Ошибка анализа"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post(
    "/analyze/async",
    response_model=TaskSubmitResponse,
    status_code=202,
    summary="Асинхронный анализ аудиозаписи",
    description=(
        "Ставит анализ в очередь и сразу возвращает task_id. "
        "Статус и результат — через GET /tasks/{task_id}."
    ),
    tags=["Анализ"],
)
async def analyze_lecture_async(
    file: UploadFile = File(..., description="Аудиофайл лекции (mp3 или wav)"),
    record_id: str = Form(..., description="Уникальный ID записи"),
    groups: List[str] = Form(..., description="Список студенческих групп"),
    lecture_date: Optional[str] = Form(
        None, description="Дата лекции (DD-MM-YYYY), по умолчанию сегодня"
    ),
):
    logger.info("POST /analyze/async record_id=%s groups=%s", record_id, groups)
    ext = os.path.splitext(file.filename or "audio.mp3")[1].lower()
    if ext not in (".mp3", ".wav"):
        raise HTTPException(400, "Поддерживаются только mp3 и wav файлы")

    with tempfile.NamedTemporaryFile(
        dir=UPLOAD_DIR, suffix=ext, delete=False
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    if lecture_date:
        try:
            date = datetime.date(
                int(lecture_date[6:10]), int(lecture_date[3:5]), int(lecture_date[0:2])
            )
        except (ValueError, IndexError):
            raise HTTPException(400, "Неверный формат даты, ожидается DD-MM-YYYY")
    else:
        date = datetime.date.today()

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": TaskStatus.pending,
        "result": None,
        "error": None,
        "created_at": datetime.datetime.utcnow(),
    }

    asyncio.create_task(_run_analysis(task_id, tmp_path, record_id, groups, date))

    return TaskSubmitResponse(task_id=task_id)


@app.get(
    "/tasks/{task_id}",
    response_model=TaskInfo,
    summary="Статус задачи анализа",
    description="Возвращает текущий статус и результат (если готов).",
    tags=["Анализ"],
)
async def get_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена")
    resp = TaskInfo(
        task_id=task_id,
        status=task["status"],
        error=task.get("error"),
    )
    if task["status"] == TaskStatus.completed and task["result"] is not None:
        resp.result = AnalysisResponse(**task["result"])
    return resp


# =============================================================================
# Endpoints — RAG: Управление лекциями
# =============================================================================


@app.post(
    "/lectures",
    response_model=AddLectureResponse,
    summary="Добавить лекцию в RAG",
    description="Чанкует текст лекции и сохраняет в PostgreSQL + Milvus для поиска.",
    tags=["Лекции"],
)
async def add_lecture(req: AddLectureRequest):
    logger.info("POST /lectures record_id=%s groups=%s", req.record_id, req.student_groups)
    parsed_date = datetime.date(
        int(req.lecture_date[6:10]), int(req.lecture_date[3:5]), int(req.lecture_date[0:2])
    )
    lecture_id = app.state.analyzer.rag.add_lecture(
        lecture_text=req.lecture_text,
        student_groups=req.student_groups,
        lecture_date=parsed_date,
        record_id=req.record_id,
    )
    return AddLectureResponse(lecture_id=lecture_id)


@app.get(
    "/lectures",
    response_model=List[LectureInfo],
    summary="Список лекций",
    description="Возвращает список лекций, опционально фильтруя по группе.",
    tags=["Лекции"],
)
async def list_lectures(
    student_group: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    lectures = app.state.analyzer.rag.list_lectures(
        student_group=student_group, limit=limit
    )
    return [
        LectureInfo(
            id=lec.id,
            record_id=lec.record_id,
            student_groups=lec.student_groups,
            lecture_date=lec.lecture_date.strftime("%d-%m-%Y"),
        )
        for lec in lectures
    ]


@app.get(
    "/lectures/{lecture_id}",
    response_model=LectureDetail,
    summary="Получить лекцию по ID",
    description="Возвращает полную информацию о лекции, включая текст.",
    tags=["Лекции"],
)
async def get_lecture(lecture_id: str):
    lecture = app.state.analyzer.rag.get_lecture(lecture_id)
    if not lecture:
        raise HTTPException(404, "Лекция не найдена")
    return LectureDetail(
        id=lecture.id,
        record_id=lecture.record_id,
        student_groups=lecture.student_groups,
        lecture_date=lecture.lecture_date.strftime("%d-%m-%Y"),
        content=lecture.content,
    )


@app.delete(
    "/lectures/{lecture_id}",
    response_model=DeleteResponse,
    summary="Удалить лекцию",
    description="Удаляет лекцию и её чанки из PostgreSQL и Milvus.",
    tags=["Лекции"],
)
async def delete_lecture(lecture_id: str):
    logger.info("DELETE /lectures/%s", lecture_id)
    deleted = app.state.analyzer.rag.delete_lecture(lecture_id)
    if not deleted:
        raise HTTPException(404, "Лекция не найдена")
    return DeleteResponse(deleted=True)


# =============================================================================
# Endpoints — RAG: Запросы
# =============================================================================


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Задать вопрос по лекциям (RAG)",
    description=(
        "Agentic RAG: переформулирует вопрос → ищет релевантные чанки → "
        "генерирует ответ на основе найденного контекста."
    ),
    tags=["RAG"],
)
async def query_rag(req: QueryRequest):
    result = app.state.analyzer.rag.query(req.question, student_group=req.student_group)
    return QueryResponse(**result)


@app.post(
    "/search",
    response_model=List[SearchResult],
    summary="Поиск по similarity",
    description="Семантический поиск по лекциям без генерации ответа. Опционально — фильтрация по группе.",
    tags=["RAG"],
)
async def search(req: SearchRequest):
    rag = app.state.analyzer.rag
    if req.student_group:
        docs = rag.search_by_group(req.query, req.student_group, k=req.k)
    else:
        docs = rag.simple_search(req.query, k=req.k)
    return [
        SearchResult(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]


@app.post(
    "/search/dates",
    response_model=List[SearchResult],
    summary="Поиск по диапазону дат",
    description="Семантический поиск в лекциях за указанный период. Опционально — фильтрация по группе.",
    tags=["RAG"],
)
async def search_by_dates(req: DateSearchRequest):
    rag = app.state.analyzer.rag
    if req.student_group:
        docs = rag.search_by_date_range_and_group(
            req.query, req.start_date, req.end_date, req.student_group, k=req.k
        )
    else:
        docs = rag.search_by_date_range(
            req.query, req.start_date, req.end_date, k=req.k
        )
    return [
        SearchResult(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]
