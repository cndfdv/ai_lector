"""RAG configuration from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    # PostgreSQL
    pg_user: str = os.getenv("POSTGRES_USER", "user")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "password")
    pg_db: str = os.getenv("POSTGRES_DB", "lectures")
    pg_host: str = os.getenv("POSTGRES_HOST", "pg")
    pg_port: str = os.getenv("POSTGRES_PORT", "5433")

    # Milvus
    milvus_host: str = os.getenv("MILVUS_HOST", "milvus")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19531"))
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "lectures")

    # LLM
    llm_url: str = os.getenv("LLM_URL", "http://10.162.1.92:1234/v1")
    llm_name: str = os.getenv("LLM_NAME", "gpt-oss-lab")
    llm_api_key: str = os.getenv("LLM_API_KEY", "not-needed")

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

    # Chunking
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "2500"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "128"))

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    max_iterations: int = int(os.getenv("RAG_MAX_ITERATIONS", "3"))

    @property
    def pg_url(self) -> str:
        """PostgreSQL connection URL."""
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )
