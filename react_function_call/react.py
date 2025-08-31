from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num:float) -> float:
    """
    A function to compute and return the triple of a number.
    :param num: a number to triple
    :return: the triple of the input number
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo").bind_tools(tools)

