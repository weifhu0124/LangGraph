from typing import Dict, Any

from agentic_rag.graph.chains.generation import generation_chain
from agentic_rag.graph.state import RagState


def generate(state: RagState) -> Dict[str, Any]:
    print("===Generate===")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
