from pprint import pprint

from dotenv import load_dotenv

from agentic_rag.graph.chains.generation import generation_chain
from agentic_rag.graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from agentic_rag.graph.chains.retrieval_grader import GradeDocuments, retrival_grader
from agentic_rag.graph.chains.router import RouteQuery, question_router
from agentic_rag.ingestion import retriever

load_dotenv()


def test_retrival_grader_answer_yes() -> None:
    question = "what is Amazon q"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrival_grader.invoke({"question": question, "document": doc_txt})

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "what is Amazon q"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrival_grader.invoke({"question": "how to make pizza", "document": doc_txt})

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "what is Amazon q"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    pprint(generation)
    assert generation is not None


def test_hallucination_grader_answer_yes() -> None:
    question = "what is Amazon q"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"question": question, "context": docs})
    res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": generation})

    assert res.binary_score == "yes"


def test_hallucination_grader_answer_no() -> None:
    question = "what is Amazon q"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": "we make a dough for pizza"})

    assert res.binary_score == "no"


def test_router_to_vector_store() -> None:
    question = "what is Amazon q"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "How to make a pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
