from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agentic_rag.graph.chains.answer_grader import answer_grader
from agentic_rag.graph.chains.hallucination_grader import hallucination_grader
from agentic_rag.graph.chains.router import RouteQuery, question_router
from agentic_rag.graph.constants import WEBSEARCH, GENERATE, RETRIEVE, GRADE_DOCUMENTS
from agentic_rag.graph.nodes import retrieve, grade_documents, web_search, generate
from agentic_rag.graph.state import RagState

load_dotenv()


def should_generate(state: RagState) -> str:
    """
    Assess if the documents are all relevant. If not, use web search to find additional information.
    Otherwise, generate the answer
    """
    print("===Assess graded documents===")

    if state["web_search"]:
        return WEBSEARCH
    else:
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: RagState) -> str:
    """
    Self-RAG reflect on the documents and answer generated to see if there is hallucination
    or answer did not address the question
    """
    print("===Check hallucination===")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})

    if score.binary_score == "yes":
        print("===Generation is grounded in / supported by documents===")
        print("===Grade generation vs question")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score == "yes":
            print("===Decision: Generation addresses question===")
            return "useful"
        else:
            print("===Decision: Generation is not addressing the question, regenerating")
            return "not useful"
    else:
        print("===Decision: Generation is not grounded in document, regenerating")
        return "not supported"


def route_question(state: RagState) -> str:
    """
    Route question to web search or RAG vector store
    """
    print("===Route question===")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource.lower() == "websearch":
        return WEBSEARCH
    else:
        return RETRIEVE


workflow = StateGraph(RagState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_conditional_entry_point(route_question, {
    WEBSEARCH: WEBSEARCH,
    RETRIEVE: RETRIEVE
})
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, should_generate, {
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE
})

workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {
    "not supported": GENERATE,
    "useful": END,
    "not useful": WEBSEARCH
})
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

rag = workflow.compile()
rag.get_graph().draw_mermaid_png(output_file_path="rag.png")
