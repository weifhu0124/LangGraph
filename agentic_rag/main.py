from dotenv import load_dotenv

from agentic_rag.graph.graph import rag

load_dotenv()

if __name__ == "__main__":
    print("==VECTOR STORE===")
    question = "Compare and contrast Amazon Q Business and Amazon Q developer"
    res = rag.invoke(input={"question": question})
    print(res)

    print("==Web Search===")
    question = "How to make a pizza"
    res = rag.invoke(input={"question": question})
    print(res)
