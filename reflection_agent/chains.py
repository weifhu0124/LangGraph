from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()


"""
A reflection agent on LinkedIn profile.
"""


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a LinkedIn expert grading a LinkedIn About Section. "
         "Generate critique and recommendations for the user's LinkedIn summary. Always provide detailed recommendations, "
         "including requests for length, professionalism, style, profile discoverability etc"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a LinkedIn expert taked with writing excellent LinkedIn About section. "
         "Generate a LinkedIn About Section that highlights user's experience and improves profile discoverability. "
         "If the user provides critique, respond with a revised version of your previous attempts." ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
