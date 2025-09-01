from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from reflexion_agent.actor import first_responder
from reflexion_agent.revisor import revisor
from reflexion_agent.tool_executor import execute_tools

load_dotenv()

MAX_ITERATION = 4 # two revise cycles

# Define the state schema (must include `messages`)

# Define the state schema (must include `messages`)
class GraphState(TypedDict):
    messages: List[BaseMessage]


# --- Node wrappers (convert function → state updates) ---
def draft_node(state: GraphState) -> GraphState:
    res: BaseMessage = first_responder.invoke(state["messages"])
    return {"messages": state["messages"] + [res]}


def execute_tools_node(state: GraphState) -> GraphState:
    res = execute_tools.invoke(state["messages"])
    return {"messages": state["messages"] + res}


def revise_node(state: GraphState) -> GraphState:
    res: BaseMessage = revisor.invoke(state["messages"])
    return {"messages": state["messages"] + [res]}


builder = StateGraph(GraphState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools_node)
builder.add_node("revise", revise_node)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: GraphState) -> str:
    if len(state["messages"]) > MAX_ITERATION:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop, {
    END: END,
    "execute_tools": "execute_tools"
})
builder.set_entry_point("draft")
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflexion.png")



if __name__ == "__main__":
    res = graph.invoke({
        "messages": [HumanMessage(
            content="Compare and contrast stock picking strategies, especially between buy & hold and invest in active funds")]
    })

    stock_essay = res["messages"][-1].tool_calls[-1]['args']["answer"]
    print(stock_essay)
    # write result to the file result.txt
    with open("result.txt", "w") as f:
        f.write(stock_essay)
