"""Shared RAG components — used by both the API server and the ingest script."""

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LM_STUDIO_URL, EMBEDDING_MODEL, CHROMA_DIR, PROMPT_TEMPLATE


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vectorstore():
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embedding_function(),
    )


def build_qa_chain(vectorstore=None):
    if vectorstore is None:
        vectorstore = load_vectorstore()

    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="not-needed",
        temperature=0.7,
        max_tokens=512,
    )

    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    answer_chain = prompt | llm | StrOutputParser()

    return answer_chain, retriever
