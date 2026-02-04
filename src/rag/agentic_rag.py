"""Agentic RAG implementation using LangGraph."""

from typing import List, Optional, TypedDict

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class ReasoningOutputParser(BaseOutputParser[str]):
    """Output parser that handles reasoning mode responses."""

    def parse(self, text) -> str:
        """Extract text from LLM response (handles reasoning mode)."""
        # If it's already a string, return it
        if isinstance(text, str):
            return text

        # If it's an AIMessage with content
        content = getattr(text, "content", text)

        # If reasoning is enabled, content is a list of dicts
        if isinstance(content, list):
            for item in reversed(content):
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

from .config import RAGConfig
from .prompts import (
    ANSWER_GRADER_PROMPT,
    GENERATION_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    QUERY_REWRITER_PROMPT,
    RELEVANCE_GRADER_PROMPT,
)
from .retriever import RAGRetriever


class AgentState(TypedDict):
    """State for the agentic RAG workflow."""

    question: str
    rewritten_question: Optional[str]
    documents: List[str]
    document_metadata: List[dict]
    generation: Optional[str]
    relevance_grade: Optional[str]  # "relevant" | "irrelevant"
    hallucination_grade: Optional[str]  # "grounded" | "hallucinated"
    answer_grade: Optional[str]  # "useful" | "not_useful"
    iteration: int
    max_iterations: int


class AgenticRAG:
    """Agentic RAG with self-reflection using LangGraph."""

    def __init__(self, config: RAGConfig, retriever: RAGRetriever):
        """Initialize agentic RAG.

        Args:
            config: RAG configuration.
            retriever: RAG retriever instance.
        """
        self.config = config
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=config.llm_name,
            base_url=config.llm_url,
            api_key=config.llm_api_key,
            temperature=0,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("check_hallucination", self._check_hallucination)
        workflow.add_node("grade_answer", self._grade_answer)

        # Define edges
        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional: after grading documents
        workflow.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
            },
        )

        workflow.add_edge("generate", "check_hallucination")

        # Conditional: after hallucination check
        workflow.add_conditional_edges(
            "check_hallucination",
            self._route_after_hallucination,
            {
                "grade_answer": "grade_answer",
                "regenerate": "generate",
            },
        )

        # Conditional: after answer grading
        workflow.add_conditional_edges(
            "grade_answer",
            self._route_after_answer_grade,
            {
                "end": END,
                "rewrite": "rewrite_query",
            },
        )

        return workflow.compile()

    # Node implementations
    def _rewrite_query(self, state: AgentState) -> dict:
        """Rewrite the user question for better retrieval."""
        prompt = ChatPromptTemplate.from_template(QUERY_REWRITER_PROMPT)
        chain = prompt | self.llm | ReasoningOutputParser()
        rewritten = chain.invoke({"question": state["question"]})
        return {
            "rewritten_question": rewritten.strip(),
            "iteration": state.get("iteration", 0) + 1,
        }

    def _retrieve(self, state: AgentState) -> dict:
        """Retrieve relevant documents."""
        query = state.get("rewritten_question") or state["question"]
        docs = self.retriever.semantic_search(query, k=self.config.top_k)
        return {
            "documents": [doc.page_content for doc in docs],
            "document_metadata": [doc.metadata for doc in docs],
        }

    def _grade_documents(self, state: AgentState) -> dict:
        """Grade document relevance."""
        prompt = ChatPromptTemplate.from_template(RELEVANCE_GRADER_PROMPT)
        chain = prompt | self.llm | ReasoningOutputParser()

        relevant_docs = []
        relevant_metadata = []
        for i, doc in enumerate(state["documents"]):
            grade = chain.invoke({
                "question": state["question"],
                "document": doc,
            }).strip().lower()
            if grade == "relevant":
                relevant_docs.append(doc)
                if i < len(state["document_metadata"]):
                    relevant_metadata.append(state["document_metadata"][i])

        return {
            "documents": relevant_docs,
            "document_metadata": relevant_metadata,
            "relevance_grade": "relevant" if relevant_docs else "irrelevant",
        }

    def _generate(self, state: AgentState) -> dict:
        """Generate answer from documents."""
        context = "\n\n---\n\n".join(state["documents"])
        prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
        chain = prompt | self.llm | ReasoningOutputParser()

        generation = chain.invoke({
            "context": context,
            "question": state["question"],
        })
        return {"generation": generation}

    def _check_hallucination(self, state: AgentState) -> dict:
        """Check if generation is grounded in documents."""
        prompt = ChatPromptTemplate.from_template(HALLUCINATION_GRADER_PROMPT)
        chain = prompt | self.llm | ReasoningOutputParser()

        grade = chain.invoke({
            "documents": "\n\n".join(state["documents"]),
            "generation": state["generation"],
        }).strip().lower()

        return {"hallucination_grade": grade}

    def _grade_answer(self, state: AgentState) -> dict:
        """Grade if answer is useful."""
        prompt = ChatPromptTemplate.from_template(ANSWER_GRADER_PROMPT)
        chain = prompt | self.llm | ReasoningOutputParser()

        grade = chain.invoke({
            "question": state["question"],
            "generation": state["generation"],
        }).strip().lower()

        return {"answer_grade": grade}

    # Routing functions
    def _route_after_grading(self, state: AgentState) -> str:
        """Route after document grading."""
        if state["relevance_grade"] == "relevant":
            return "generate"
        if state["iteration"] < state["max_iterations"]:
            return "rewrite"
        return "generate"  # Force generation even with poor docs

    def _route_after_hallucination(self, state: AgentState) -> str:
        """Route after hallucination check."""
        if state["hallucination_grade"] == "grounded":
            return "grade_answer"
        if state["iteration"] < state["max_iterations"]:
            return "regenerate"
        return "grade_answer"  # Accept even if hallucinated after max tries

    def _route_after_answer_grade(self, state: AgentState) -> str:
        """Route after answer grading."""
        if state["answer_grade"] == "useful":
            return "end"
        if state["iteration"] < state["max_iterations"]:
            return "rewrite"
        return "end"  # Return whatever we have

    # Public API
    def query(self, question: str, max_iterations: int = 3) -> dict:
        """Answer a question using agentic RAG.

        Args:
            question: User question.
            max_iterations: Maximum self-correction iterations.

        Returns:
            Dict with answer, sources, iterations, and grades.
        """
        initial_state: AgentState = {
            "question": question,
            "rewritten_question": None,
            "documents": [],
            "document_metadata": [],
            "generation": None,
            "relevance_grade": None,
            "hallucination_grade": None,
            "answer_grade": None,
            "iteration": 0,
            "max_iterations": max_iterations,
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result["generation"],
            "sources": result["document_metadata"],
            "rewritten_question": result["rewritten_question"],
            "iterations": result["iteration"],
            "grades": {
                "relevance": result["relevance_grade"],
                "hallucination": result["hallucination_grade"],
                "answer": result["answer_grade"],
            },
        }
