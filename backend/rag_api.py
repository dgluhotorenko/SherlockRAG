"""FastAPI server that exposes the RAG chain as a REST endpoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from rag_chain import build_qa_chain

app = FastAPI(title="Sherlock RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


answer_chain, retriever = build_qa_chain()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    docs = retriever.invoke(request.question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = answer_chain.invoke({"context": context, "question": request.question})
    return QueryResponse(
        answer=answer,
        sources=list({doc.metadata.get("source", "N/A") for doc in docs}),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
