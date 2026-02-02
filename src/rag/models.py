"""SQLAlchemy models for lecture storage."""

from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Lecture(Base):
    """Lecture metadata and content."""

    __tablename__ = "lectures"

    id = Column(String(36), primary_key=True)
    record_id = Column(String(255), unique=True, nullable=True)
    student_group = Column(Text, nullable=False)
    lecture_date = Column(Date, nullable=False)
    content = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship(
        "LectureChunk",
        back_populates="lecture",
        cascade="all, delete-orphan",
    )


class LectureChunk(Base):
    """Lecture chunk for tracking Milvus entries."""

    __tablename__ = "lecture_chunks"

    chunk_id = Column(String(255), primary_key=True)
    lecture_id = Column(
        String(36),
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    lecture = relationship("Lecture", back_populates="chunks")
