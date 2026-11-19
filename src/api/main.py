# src/api/main.py
from fastapi import FastAPI, HTTPException

from src.api.schemas import QueryRequest, QueryResponse
from src.generation.rag import answer_question

app = FastAPI(
    title="RAG Eval Engine API",
    description="Local RAG pipeline over technical documentation using PostgreSQL + pgvector and Ollama.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest) -> QueryResponse:
    try:
        result = answer_question(query=payload.question, top_k=payload.top_k)
        return QueryResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc