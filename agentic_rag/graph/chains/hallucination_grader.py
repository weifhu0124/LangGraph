from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


llm = ChatOpenAI(temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination check on retrieved documents"""

    binary_score: str = Field(
        description="Documents are not hallucinated, either 'yes' or 'no' "
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)


system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
