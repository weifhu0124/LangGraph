from typing import Any, Dict

from agentic_rag.graph.state import RagState
from agentic_rag.ingestion import retriever


def retrieve(state: RagState) -> Dict[str, Any]:
    """
    Retrieve documents from vector store based on the question.
    """
    print("===Retrieve===")
    question = state["question"]

    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}