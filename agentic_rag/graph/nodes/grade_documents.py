from typing import Dict, Any

from agentic_rag.graph.chains.retrieval_grader import retrival_grader
from agentic_rag.graph.state import RagState


def grade_documents(state: RagState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("===Check document relevance to question===")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrival_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score.lower() == "yes":
            print("---DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---DOCUMENT NOT RELEVANT---")
            web_search = True
    return {"documents": filtered_docs, "web_search": web_search, "question": question}
