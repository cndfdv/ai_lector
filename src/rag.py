import os
import uuid

from chonkie import Pipeline
from chonkie.genie import OpenAIGenie
from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sqlalchemy import Column, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Lecture(Base):
    __tablename__ = "lectures"
    id = Column(String, primary_key=True)
    content = Column(Text)


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class RAG:
    def __init__(self):
        milvus_host = os.getenv("MILVUS_HOST", "milvus")
        milvus_port = int(os.getenv("MILVUS_PORT", "19531"))
        milvus_collection = os.getenv("MILVUS_COLLECTION", "lectures")

        pg_user = os.getenv("POSTGRES_USER", "user")
        pg_password = os.getenv("POSTGRES_PASSWORD", "password")
        pg_db = os.getenv("POSTGRES_DB", "lectures")
        pg_host = os.getenv("POSTGRES_HOST", "pg")
        pg_port = os.getenv("POSTGRES_PORT", "5433")
        pg_url = (
            f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )
        self.llm = ChatOpenAI(
            model="gpt-oss-lab",
            temperature=0,
            base_url="http://10.162.1.92:1234/v1",
            api_key="not-needed",
            reasoning={"effort": "high", "summary": "auto"},
        )
        self.genie = OpenAIGenie(
            model="gpt-oss-lab",
            base_url="http://10.162.1.92:1234/v1",
            api_key="not-needed",
        )
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=milvus_collection,
            connection_args={
                "host": milvus_host,
                "port": milvus_port,
            },
            drop_old=False,
        )

        self.engine = create_engine(pg_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_lecture(self, lecture_text: str):
        pipe = Pipeline().chunk_with(
            "slumber",
            genie=self.genie,
            tokenizer="gpt2",
            chunk_size=2500,
            candidate_size=2500,
            min_characters_per_chunk=128,
            verbose=False,
        )

        doc = pipe.run(texts=lecture_text)

        lecture_id = str(uuid.uuid4())

        texts, metadatas, doc_ids = [], [], []
        for i, chunk in enumerate(doc.chunks):
            chunk_text = getattr(chunk, "content", str(chunk))
            texts.append(chunk_text)
            metadatas.append({"lecture_id": lecture_id})
            doc_ids.append(f"{lecture_id}_{i}")

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=doc_ids)

        self.save_to_postgres(lecture_id, doc.content)
        return lecture_id

    def save_to_postgres(self, lecture_id: str, content: str):
        session = self.Session()
        session.add(Lecture(id=lecture_id, content=content))
        session.commit()
        session.close()

    def similarity_search(self, query, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)

    def answer_query(self, user_question: str, k: int = 5):
        if not self.llm:
            raise ValueError("LLM не указан. Передайте llm при инициализации класса.")

        chunks = self.similarity_search(user_question, k=k)
        context = "\n".join([c.page_content for c in chunks])
        prompt = f"Вопрос: {user_question}\n\nКонтекст из лекций:\n{context}\n\nДай подробный и понятный ответ:"
        return self.llm(prompt)

    def answer_query_agent(self, user_question: str, k: int = 5):
        if not self.llm:
            raise ValueError("LLM не указан. Передайте llm при инициализации класса.")

        rewrite_prompt = f"Переделай этот вопрос так, чтобы его было проще искать в базе:\n\n{user_question}\n\nПереформулированный запрос:"
        rewritten_question = self.llm.invoke(rewrite_prompt).content[1]["text"]

        chunks = self.similarity_search(rewritten_question, k=k)
        context = "\n".join([c.page_content for c in chunks])

        answer_prompt = f"Вопрос: {user_question}\n\nПереформулированный запрос: {rewritten_question}\n\nКонтекст из лекций:\n{context}\n\nДай подробный и понятный ответ:"
        return self.llm.invoke(answer_prompt).content[1]["text"]

    def get_lecture(self, lecture_id: str):
        session = self.Session()
        lecture = session.query(Lecture).filter_by(id=lecture_id).first()
        session.close()
        return lecture.content if lecture else None
