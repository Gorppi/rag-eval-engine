# src/api/schemas.py
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["How do path parameters work in FastAPI?"])
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]