from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Critique of the answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries to research information to improve the answer.",
        default=[]
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    reference: List[str] = Field(description="Citations motivating your updated answer.")
