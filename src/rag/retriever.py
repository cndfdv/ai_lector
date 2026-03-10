"""Retrieval tools for agentic RAG."""

from typing import List, Optional

from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document


class RAGRetriever:
    """Retriever with multiple search strategies."""

    def __init__(self, vector_store: Milvus):
        """Initialize retriever.

        Args:
            vector_store: Milvus vector store instance.
        """
        self.vector_store = vector_store

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant lecture content based on semantic similarity.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of relevant documents.
        """
        return self.vector_store.similarity_search(query, k=k)

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
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=f'array_contains(student_groups, "{student_group}")',
        )

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
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            k: Number of results.

        Returns:
            Relevant content from lectures in the date range.
        """
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter={
                "lecture_date": {"$gte": start_date, "$lte": end_date},
            },
        )

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
