"""Pydantic models for LLM structured outputs."""

from typing import List

from pydantic import BaseModel, Field


class QuestionsResult(BaseModel):
    """List of self-check questions for a lecture."""

    questions: List[str] = Field(
        description="10-12 вопросов для самопроверки по материалу лекции",
        min_length=10,
        max_length=15,
    )


class MindMapNode(BaseModel):
    """A node in the mind map hierarchy."""

    id: str = Field(description="Уникальный идентификатор узла")
    label: str = Field(description="Название темы или подтемы")
    children: List["MindMapNode"] = Field(
        default_factory=list, description="Дочерние узлы (подтемы)"
    )


class MindMap(BaseModel):
    """Hierarchical mind map of lecture content."""

    title: str = Field(description="Название лекции")
    nodes: List[MindMapNode] = Field(description="Основные темы лекции")


class PodcastPart(BaseModel):
    """A single part of podcast dialogue."""

    presenter: str = Field(description="Реплика ведущего")
    lector: str = Field(description="Реплика лектора")


class PodcastScript(BaseModel):
    """Complete podcast script as a list of dialogue parts."""

    parts: List[PodcastPart] = Field(
        description="Список частей диалога подкаста", min_length=1
    )


class EmotionalAnalysis(BaseModel):
    """Emotional analysis result - two adjectives describing the fragment."""

    adjective1: str = Field(description="Первое прилагательное, описывающее фрагмент")
    adjective2: str = Field(description="Второе прилагательное, описывающее фрагмент")


# Allow forward references for recursive MindMapNode
MindMapNode.model_rebuild()
