# src/retrieval/search.py
from dataclasses import dataclass

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

from src.config import settings


@dataclass
class RetrievedChunk:
    source: str
    chunk_index: int
    content: str
    similarity: float


def retrieve_chunks(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar document chunks."""
    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    query_embedding = (
        client.embeddings.create(model=settings.embedding_model, input=query)
        .data[0]
        .embedding
    )

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, chunk_index, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            source=row[0],
            chunk_index=row[1],
            content=row[2],
            similarity=round(float(row[3]), 4),
        )
        for row in rows
    ]