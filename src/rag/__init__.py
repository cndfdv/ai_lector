"""Lecture RAG module for storing and querying lecture content."""

import json
import uuid
from datetime import date
from typing import List, Optional

from chonkie import Pipeline
from chonkie.genie import OpenAIGenie
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .agentic_rag import AgenticRAG
from .config import RAGConfig
from .models import Base, Lecture, LectureChunk
from .retriever import RAGRetriever


class LectureRAG:
    """Main RAG class for lecture management and querying."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the RAG system.

        Args:
            config: RAG configuration. If None, uses default from environment.
        """
        self.config = config or RAGConfig()

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
        )

        # Initialize vector store
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.config.milvus_collection,
            connection_args={
                "host": self.config.milvus_host,
                "port": self.config.milvus_port,
            },
            drop_old=False,
        )

        # Initialize PostgreSQL
        self.engine = create_engine(self.config.pg_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Initialize chunking genie
        self.genie = OpenAIGenie(
            model=self.config.llm_name,
            base_url=self.config.llm_url,
            api_key=self.config.llm_api_key,
        )

        # Initialize retriever and agent
        self._retriever = RAGRetriever(self.vector_store)
        self._agent = AgenticRAG(self.config, self._retriever)

    def add_lecture(
        self,
        lecture_text: str,
        student_groups: List[str],
        lecture_date: date,
        record_id: Optional[str] = None,
    ) -> str:
        """Add a lecture to the RAG system.

        Args:
            lecture_text: Full lecture text.
            student_groups: List of student group identifiers.
            lecture_date: Date of the lecture.
            record_id: Optional unique record identifier.

        Returns:
            Lecture ID (UUID).
        """
        lecture_id = str(uuid.uuid4())

        # Chunk the lecture using chonkie
        pipe = Pipeline().chunk_with(
            "slumber",
            genie=self.genie,
            tokenizer="gpt2",
            chunk_size=self.config.chunk_size,
            candidate_size=self.config.chunk_size,
            min_characters_per_chunk=self.config.chunk_overlap,
            verbose=False,
        )
        doc = pipe.run(texts=lecture_text)

        # Prepare chunks for vector store
        texts: List[str] = []
        metadatas: List[dict] = []
        chunk_ids: List[str] = []

        for i, chunk in enumerate(doc.chunks):
            chunk_text = getattr(chunk, "content", str(chunk))
            chunk_id = f"{lecture_id}_{i}"

            texts.append(chunk_text)
            metadatas.append({
                "lecture_id": lecture_id,
                "student_groups": json.dumps(student_groups),
                "lecture_date": lecture_date.isoformat(),
                "chunk_index": i,
            })
            chunk_ids.append(chunk_id)

        # Add to vector store
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=chunk_ids,
        )

        # Save to PostgreSQL
        session = self.Session()
        try:
            lecture = Lecture(
                id=lecture_id,
                record_id=record_id,
                student_groups=student_groups,
                lecture_date=lecture_date,
                content=doc.content,
            )
            session.add(lecture)

            # Save chunks
            for i, chunk_text in enumerate(texts):
                chunk = LectureChunk(
                    chunk_id=chunk_ids[i],
                    lecture_id=lecture_id,
                    chunk_index=i,
                    chunk_text=chunk_text,
                )
                session.add(chunk)

            session.commit()
        finally:
            session.close()

        return lecture_id

    def query(self, question: str, student_group: Optional[str] = None) -> dict:
        """Query the RAG system: rewrite → retrieve → generate.

        Args:
            question: User question.
            student_group: Optional student group filter.

        Returns:
            Dict with answer, sources, and rewritten_question.
        """
        return self._agent.query(question, student_group=student_group)

    def simple_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Simple similarity search without agentic reasoning.

        Args:
            query: Search query.
            k: Number of results. Uses config default if not specified.

        Returns:
            List of matching documents.
        """
        k = k or self.config.top_k
        return self._retriever.semantic_search(query, k=k)

    def search_by_group(
        self,
        query: str,
        student_group: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Search lectures for a specific student group.

        Args:
            query: Search query.
            student_group: Student group identifier.
            k: Number of results.

        Returns:
            List of matching documents.
        """
        k = k or self.config.top_k
        return self._retriever.search_by_group(query, student_group, k=k)

    def search_by_date_range(
        self,
        query: str,
        start_date: str,
        end_date: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Search lectures within a date range.

        Args:
            query: Search query.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            k: Number of results.

        Returns:
            List of matching documents.
        """
        k = k or self.config.top_k
        return self._retriever.search_by_date_range(query, start_date, end_date, k=k)

    def get_lecture(self, lecture_id: str) -> Optional[Lecture]:
        """Get lecture by ID.

        Args:
            lecture_id: Lecture UUID.

        Returns:
            Lecture object or None if not found.
        """
        session = self.Session()
        try:
            return session.query(Lecture).filter_by(id=lecture_id).first()
        finally:
            session.close()

    def get_lecture_by_record_id(self, record_id: str) -> Optional[Lecture]:
        """Get lecture by record ID.

        Args:
            record_id: Recording identifier.

        Returns:
            Lecture object or None if not found.
        """
        session = self.Session()
        try:
            return session.query(Lecture).filter_by(record_id=record_id).first()
        finally:
            session.close()

    def list_lectures(
        self,
        student_group: Optional[str] = None,
        limit: int = 100,
    ) -> List[Lecture]:
        """List lectures, optionally filtered by group.

        Args:
            student_group: Filter by student group.
            limit: Maximum number of results.

        Returns:
            List of Lecture objects.
        """
        session = self.Session()
        try:
            query = session.query(Lecture)
            if student_group:
                query = query.filter(Lecture.student_groups.contains([student_group]))
            return query.order_by(Lecture.lecture_date.desc()).limit(limit).all()
        finally:
            session.close()

    def delete_lecture(self, lecture_id: str) -> bool:
        """Delete lecture and its chunks from both databases.

        Args:
            lecture_id: Lecture UUID.

        Returns:
            True if deleted, False if not found.
        """
        session = self.Session()
        try:
            lecture = session.query(Lecture).filter_by(id=lecture_id).first()
            if not lecture:
                return False

            # Delete from vector store
            chunk_ids = [c.chunk_id for c in lecture.chunks]
            if chunk_ids:
                try:
                    self.vector_store.delete(chunk_ids)
                except Exception:
                    pass  # Milvus may not support delete on some configs

            # Delete from PostgreSQL (cascades to chunks)
            session.delete(lecture)
            session.commit()
            return True
        finally:
            session.close()


# Export main class
__all__ = ["LectureRAG", "RAGConfig"]
