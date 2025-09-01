from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Any


load_dotenv()


urls = [
    "https://aws.amazon.com/bedrock/agentcore/",
    "https://aws.amazon.com/q/business/",
    "https://aws.amazon.com/q/developer/build/",
]


def load_documents() -> List[Any]:
    # load documents from web
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def split_text(docs_list: List[Any]) -> List[Document]:
    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    return text_splitter.split_documents(docs_list)


def ingest() -> None:
    documents = load_documents()
    docs_list = split_text(documents)

    # check if directory ./.chroma_db exists
    import os
    if not os.path.exists("./.chroma_db"):
        # ingest into vector store
        Chroma.from_documents(
            documents=docs_list,
            collection_name="rag-aws-agents-chroma",
            embedding=OpenAIEmbeddings(),
            persist_directory="./.chroma_db",
        )

ingest()
# create a retriever
retriever = Chroma(
    collection_name="rag-aws-agents-chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./.chroma_db",
).as_retriever()
