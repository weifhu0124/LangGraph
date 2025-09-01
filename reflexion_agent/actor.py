import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from reflexion_agent.schemas import AnswerQuestion


load_dotenv()


llm = ChatOpenAI(model="gpt-4-turbo-preview")
# return function call result as a dictionary
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
                Current time: {time}

                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Must provide 1-3 search queries to research information and improve your answer.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format. You must provide search queries"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Generate a ~250 word answer to the user's question"
)

first_responder = (first_responder_prompt_template
                   | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion"))

