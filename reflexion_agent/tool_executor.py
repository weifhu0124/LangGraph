from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from reflexion_agent.schemas import AnswerQuestion, ReviseAnswer

load_dotenv()


tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: List[str], **kwargs):
    """Run the generated queries"""
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode([
    StructuredTool.from_function(
        func=run_queries,
        name=AnswerQuestion.__name__,
    ),
    StructuredTool.from_function(
        func=run_queries,
        name=ReviseAnswer.__name__,
    )],
    messages_key="messages"
)