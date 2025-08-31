from typing import List, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, StateGraph, MessagesState

from chains import reflection_chain, generation_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"


# Define the state schema (must include `messages`)
class GraphState(TypedDict):
    messages: List[BaseMessage]


def should_continue(state: GraphState) -> str:
    if len(state["messages"]) > 3:
        return END
    return REFLECT


def generation_node(state: GraphState) -> GraphState:
    res = generation_chain.invoke({"messages": state["messages"]})
    # return new messages appended
    return {"messages": state["messages"] + [res]}


def reflection_node(state: GraphState) -> GraphState:
    res = reflection_chain.invoke({"messages": state["messages"]})
    # trick LLM to think user is sending the critique message
    return {"messages": state["messages"] + [HumanMessage(content=res.content)]}


# Build graph with StateGraph instead of MessageGraph
builder = StateGraph(GraphState)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        END: END,
        REFLECT: REFLECT
    }
)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="reflection.png")


if __name__ == "__main__":
    about_section = """ Make this about section better: 
    At Amazon Web Services (AWS), contributed to transforming traditional search systems into advanced AI-powered data discovery services, optimizing content accessibility for internal stakeholders. Led the technical vision that kicked off the creation of an intelligent data curation and discovery service, enhancing content accessibility and usability. Designed and implemented scalable proxy APIs for large language models (LLMs), enabling seamless interactions for over 1 million employees and 25,000+ daily active users. Enhanced system performance by reducing latency and minimizing redundant call volumes. 
    Earned a Master's degree in Computer Science from the University of California, San Diego, with a focus on machine learning and neural networks. Leveraged this expertise to lead impactful projects, including conversational AI interfaces and similarity model enhancements using AWS SageMaker. Fluent in Java, Python, React and Typescript (CDK). Passionate about advancing AI-driven solutions to improve efficiency and user experience for global-scale enterprises.
    """

    inputs = HumanMessage(content=about_section)
    res = graph.invoke({"messages": [inputs]})
    print(res["messages"][-1].content)
