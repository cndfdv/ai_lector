"""Retrieval tools for agentic RAG."""

import json
import re
from datetime import date
from typing import List

from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document

# Pattern for safe student_group values: letters, digits, dots, hyphens, underscores, spaces
_SAFE_GROUP_RE = re.compile(r"^[a-zA-Zа-яА-ЯёЁ0-9_\-. ]+$")


def _validate_group(student_group: str) -> str:
    """Validate student_group to prevent Milvus expression injection."""
    if not _SAFE_GROUP_RE.match(student_group):
        raise ValueError(f"Недопустимые символы в student_group: {student_group!r}")
    return student_group


def _validate_date(value: str, field_name: str) -> str:
    """Validate date string format (DD-MM-YYYY) and convert to YYYY-MM-DD for Milvus."""
    if len(value) != 10:
        raise ValueError(f"Неверный формат даты {field_name}: {value!r}, ожидается DD-MM-YYYY")
    try:
        parsed = date(int(value[6:10]), int(value[3:5]), int(value[0:2]))
    except (ValueError, IndexError):
        raise ValueError(f"Неверный формат даты {field_name}: {value!r}, ожидается DD-MM-YYYY")
    return parsed.isoformat()


class RAGRetriever:
    """Retriever with multiple search strategies."""

    def __init__(self, vector_store: Milvus):
        """Initialize retriever.

        Args:
            vector_store: Milvus vector store instance.
        """
        self.vector_store = vector_store

    @staticmethod
    def _deserialize_metadata(docs: List[Document]) -> List[Document]:
        """Deserialize JSON-encoded metadata fields from Milvus."""
        for doc in docs:
            sg = doc.metadata.get("student_groups")
            if isinstance(sg, str):
                try:
                    doc.metadata["student_groups"] = json.loads(sg)
                except (json.JSONDecodeError, TypeError):
                    pass
        return docs

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant lecture content based on semantic similarity.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of relevant documents.
        """
        docs = self.vector_store.similarity_search(query, k=k)
        return self._deserialize_metadata(docs)

    def search_by_group(
        self,
        query: str,
        student_group: str,
        k: int = 5,
    ) -> List[Document]:
        """Search lectures for a specific student group.

        Args:
            query: Search query.
            student_group: Student group identifier.
            k: Number of results.

        Returns:
            Relevant content from the specified group's lectures.
        """
        safe_group = _validate_group(student_group)
        docs = self.vector_store.similarity_search(
            query,
            k=k,
            expr=f'student_groups like "%{safe_group}%"',
        )
        return self._deserialize_metadata(docs)

    def search_by_date_range(
        self,
        query: str,
        start_date: str,
        end_date: str,
        k: int = 5,
    ) -> List[Document]:
        """Search lectures within a date range.

        Args:
            query: Search query.
            start_date: Start date (DD-MM-YYYY).
            end_date: End date (DD-MM-YYYY).
            k: Number of results.

        Returns:
            Relevant content from lectures in the date range.
        """
        safe_start = _validate_date(start_date, "start_date")
        safe_end = _validate_date(end_date, "end_date")
        docs = self.vector_store.similarity_search(
            query,
            k=k,
            expr=f'lecture_date >= "{safe_start}" and lecture_date <= "{safe_end}"',
        )
        return self._deserialize_metadata(docs)

    def search_by_date_range_and_group(
        self,
        query: str,
        start_date: str,
        end_date: str,
        student_group: str,
        k: int = 5,
    ) -> List[Document]:
        """Search lectures within a date range for a specific student group.

        Args:
            query: Search query.
            start_date: Start date (DD-MM-YYYY).
            end_date: End date (DD-MM-YYYY).
            student_group: Student group identifier.
            k: Number of results.

        Returns:
            Relevant content from the specified group's lectures in the date range.
        """
        safe_start = _validate_date(start_date, "start_date")
        safe_end = _validate_date(end_date, "end_date")
        safe_group = _validate_group(student_group)
        expr = (
            f'lecture_date >= "{safe_start}" and lecture_date <= "{safe_end}"'
            f' and student_groups like "%{safe_group}%"'
        )
        docs = self.vector_store.similarity_search(query, k=k, expr=expr)
        return self._deserialize_metadata(docs)

    def format_documents(self, docs: List[Document]) -> str:
        """Format documents for context.

        Args:
            docs: List of documents.

        Returns:
            Formatted string with document contents.
        """
        results = []
        for doc in docs:
            lecture_id = doc.metadata.get("lecture_id", "unknown")
            group = doc.metadata.get("student_groups", "unknown")
            date = doc.metadata.get("lecture_date", "unknown")
            header = f"[Лекция: {lecture_id}, Группа: {group}, Дата: {date}]"
            results.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(results)
