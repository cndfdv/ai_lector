"""Simple RAG pipeline: rewrite → retrieve → generate."""

from typing import List, Optional, TypedDict

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import RAGConfig
from .prompts import GENERATION_PROMPT, QUERY_REWRITER_PROMPT
from .retriever import RAGRetriever


class ReasoningOutputParser(BaseOutputParser[str]):
    """Output parser that handles reasoning mode responses."""

    def parse(self, text) -> str:
        if isinstance(text, str):
            return text

        content = getattr(text, "content", text)

        if isinstance(content, list):
            for item in reversed(content):
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)


class AgentState(TypedDict):
    """State for the RAG workflow."""

    question: str
    student_group: Optional[str]
    rewritten_question: Optional[str]
    documents: List[str]
    document_metadata: List[dict]
    generation: Optional[str]


class AgenticRAG:
    """Simple RAG: rewrite query → retrieve top-k → generate answer."""

    def __init__(self, config: RAGConfig, retriever: RAGRetriever):
        self.config = config
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=config.llm_name,
            base_url=config.llm_url,
            api_key=config.llm_api_key,
            reasoning={"effort": "high", "summary": None},
            temperature=0,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)

        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _rewrite_query(self, state: AgentState) -> dict:
        """Rewrite the user question for better retrieval."""
        prompt = ChatPromptTemplate.from_template(QUERY_REWRITER_PROMPT)
        chain = prompt | self.llm | ReasoningOutputParser()
        rewritten = chain.invoke({"question": state["question"]})
        return {"rewritten_question": rewritten.strip()}

    def _retrieve(self, state: AgentState) -> dict:
        """Retrieve top-k relevant documents, optionally filtered by student group."""
        query = state.get("rewritten_question") or state["question"]
        student_group = state.get("student_group")
        if student_group:
            docs = self.retriever.search_by_group(query, student_group, k=self.config.top_k)
        else:
            docs = self.retriever.semantic_search(query, k=self.config.top_k)
        return {
            "documents": [doc.page_content for doc in docs],
            "document_metadata": [doc.metadata for doc in docs],
        }

    def _generate(self, state: AgentState) -> dict:
        """Generate answer from retrieved documents."""
        context = "\n\n---\n\n".join(state["documents"])
        prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
        chain = prompt | self.llm | ReasoningOutputParser()

        generation = chain.invoke({
            "context": context,
            "question": state["question"],
        })
        return {"generation": generation}

    def query(self, question: str, student_group: Optional[str] = None) -> dict:
        """Answer a question using RAG.

        Args:
            question: User question.
            student_group: Optional student group filter.

        Returns:
            Dict with answer, sources, and rewritten_question.
        """
        initial_state: AgentState = {
            "question": question,
            "student_group": student_group,
            "rewritten_question": None,
            "documents": [],
            "document_metadata": [],
            "generation": None,
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result["generation"],
            "sources": result["document_metadata"],
            "rewritten_question": result["rewritten_question"],
        }
