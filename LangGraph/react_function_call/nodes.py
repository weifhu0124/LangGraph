from dotenv import load_dotenv
from langgraph.graph import MessagesState # track the human and AI messages
from langgraph.prebuilt import ToolNode

from react import llm, tools


load_dotenv()


SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}


tool_node = ToolNode(tools)
