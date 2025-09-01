"""
Graph state of Agentic Rag
"""
from typing import TypedDict


class RagState(TypedDict):
    """
    Represent state of our RAG system

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: list[str]
