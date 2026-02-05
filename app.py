"""FastAPI application for AI Lector — lecture analysis and RAG service."""

import datetime
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.rag import LectureRAG
from src.rec_analyzer import LectureAnalyzer

# =============================================================================
# Pydantic Models — Request
# =============================================================================


class AddLectureRequest(BaseModel):
    """Запрос на добавление лекции в RAG."""

    lecture_text: str = Field(..., description="Полный текст лекции")
    student_group: str = Field(..., description="Идентификатор студенческой группы")
    lecture_date: datetime.date = Field(..., description="Дата лекции (YYYY-MM-DD)")
    record_id: Optional[str] = Field(None, description="Уникальный ID записи")
    abstract: Optional[str] = Field(None, description="Конспект лекции (markdown)")


class QueryRequest(BaseModel):
    """Запрос к RAG системе."""

    question: str = Field(..., description="Вопрос по лекциям")


class SearchRequest(BaseModel):
    """Запрос на поиск по similarity."""

    query: str = Field(..., description="Поисковый запрос")
    k: int = Field(5, ge=1, le=50, description="Количество результатов")


class GroupSearchRequest(BaseModel):
    """Запрос на поиск по группе."""

    query: str = Field(..., description="Поисковый запрос")
    student_group: str = Field(..., description="Идентификатор студенческой группы")
    k: int = Field(5, ge=1, le=50, description="Количество результатов")


class DateSearchRequest(BaseModel):
    """Запрос на поиск по диапазону дат."""

    query: str = Field(..., description="Поисковый запрос")
    start_date: str = Field(..., description="Начальная дата (YYYY-MM-DD)")
    end_date: str = Field(..., description="Конечная дата (YYYY-MM-DD)")
    k: int = Field(5, ge=1, le=50, description="Количество результатов")


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
        ..., description="Таймлайн лекции с эмоциями"
    )
    questions: List[str] = Field(
        ..., description="Вопросы для самопроверки (10-12 шт)"
    )
    podcast: str = Field(..., description="Путь к сгенерированному подкасту (mp3)")


class LectureInfo(BaseModel):
    """Краткая информация о лекции."""

    id: str = Field(..., description="UUID лекции")
    record_id: Optional[str] = Field(None, description="ID записи")
    student_group: str = Field(..., description="Студенческая группа")
    lecture_date: datetime.date = Field(..., description="Дата лекции")
    abstract: Optional[str] = Field(None, description="Конспект")


class LectureDetail(BaseModel):
    """Полная информация о лекции."""

    id: str
    record_id: Optional[str] = None
    student_group: str
    lecture_date: datetime.date
    content: str = Field(..., description="Полный текст лекции")
    abstract: Optional[str] = None


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
# Application
# =============================================================================

UPLOAD_DIR = "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка ML моделей при старте, очистка при остановке."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.state.analyzer = LectureAnalyzer()
    app.state.rag = LectureRAG()
    yield


app = FastAPI(
    title="AI Lector",
    description=(
        "Сервис анализа лекций: транскрипция, конспект, майнд-карта, "
        "вопросы для самопроверки, подкаст, RAG по базе лекций."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


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
    group: str = Form(..., description="Идентификатор студенческой группы"),
    lecture_date: Optional[str] = Form(
        None, description="Дата лекции (YYYY-MM-DD), по умолчанию сегодня"
    ),
):
    ext = os.path.splitext(file.filename or "audio.mp3")[1].lower()
    if ext not in (".mp3", ".wav"):
        raise HTTPException(400, "Поддерживаются только mp3 и wav файлы")

    with tempfile.NamedTemporaryFile(
        dir=UPLOAD_DIR, suffix=ext, delete=False
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    date = (
        datetime.date.fromisoformat(lecture_date)
        if lecture_date
        else datetime.date.today()
    )

    try:
        result = app.state.analyzer.process(tmp_path, record_id, group, date)
    except FileNotFoundError:
        raise HTTPException(404, "Аудиофайл не найден")
    except Exception as e:
        raise HTTPException(500, f"Ошибка анализа: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return AnalysisResponse(**result)


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
    lecture_id = app.state.rag.add_lecture(
        lecture_text=req.lecture_text,
        student_group=req.student_group,
        lecture_date=req.lecture_date,
        record_id=req.record_id,
        abstract=req.abstract,
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
    limit: int = 100,
):
    lectures = app.state.rag.list_lectures(
        student_group=student_group, limit=limit
    )
    return [
        LectureInfo(
            id=lec.id,
            record_id=lec.record_id,
            student_group=lec.student_group,
            lecture_date=lec.lecture_date,
            abstract=lec.abstract,
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
    lecture = app.state.rag.get_lecture(lecture_id)
    if not lecture:
        raise HTTPException(404, "Лекция не найдена")
    return LectureDetail(
        id=lecture.id,
        record_id=lecture.record_id,
        student_group=lecture.student_group,
        lecture_date=lecture.lecture_date,
        content=lecture.content,
        abstract=lecture.abstract,
    )


@app.delete(
    "/lectures/{lecture_id}",
    response_model=DeleteResponse,
    summary="Удалить лекцию",
    description="Удаляет лекцию и её чанки из PostgreSQL и Milvus.",
    tags=["Лекции"],
)
async def delete_lecture(lecture_id: str):
    deleted = app.state.rag.delete_lecture(lecture_id)
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
    result = app.state.rag.query(req.question)
    return QueryResponse(**result)


@app.post(
    "/search",
    response_model=List[SearchResult],
    summary="Поиск по similarity",
    description="Семантический поиск по всем лекциям без генерации ответа.",
    tags=["RAG"],
)
async def search(req: SearchRequest):
    docs = app.state.rag.simple_search(req.query, k=req.k)
    return [
        SearchResult(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]


@app.post(
    "/search/group",
    response_model=List[SearchResult],
    summary="Поиск по группе",
    description="Семантический поиск в лекциях конкретной студенческой группы.",
    tags=["RAG"],
)
async def search_by_group(req: GroupSearchRequest):
    docs = app.state.rag.search_by_group(req.query, req.student_group, k=req.k)
    return [
        SearchResult(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]


@app.post(
    "/search/dates",
    response_model=List[SearchResult],
    summary="Поиск по диапазону дат",
    description="Семантический поиск в лекциях за указанный период.",
    tags=["RAG"],
)
async def search_by_dates(req: DateSearchRequest):
    docs = app.state.rag.search_by_date_range(
        req.query, req.start_date, req.end_date, k=req.k
    )
    return [
        SearchResult(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]
