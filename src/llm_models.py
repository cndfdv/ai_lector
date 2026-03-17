"""Pydantic models for LLM structured outputs."""

from typing import List

from pydantic import BaseModel, Field


class Question(BaseModel):
    question: str = Field(description="Один вопрос для самопроверки", min_length=1)


class QuestionsResult(BaseModel):
    questions: List[Question] = Field(
        description="12 вопросов для самопроверки по конспекту лекции",
        min_items=10,
        max_items=15,
    )


class PodcastPart(BaseModel):
    """A single part of podcast dialogue."""

    presenter: str = Field(description="Реплика ведущего")
    lector: str = Field(description="Реплика лектора")


class PodcastScript(BaseModel):
    """Complete podcast script as a list of dialogue parts."""

    parts: List[PodcastPart] = Field(
        description="Список частей диалога подкаста", min_length=1
    )
