from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from agentic_rag.graph.state import RagState


load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: RagState) -> Dict[str, Any]:
    print("===Web Search===")
    question = state["question"]
    documents = state["documents"] if "documents" in state.keys() else []

    tavily_results = web_search_tool.invoke({"query": question})
    joined_results = "\n".join([tavily_result["content"] for tavily_result in tavily_results['results']])
    web_results = Document(page_content=joined_results)
    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search({"question": "What is the Amazon Q?", "documents": []})
