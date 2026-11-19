# src/generation/rag.py
from openai import OpenAI

from src.config import settings
from src.retrieval.search import RetrievedChunk, retrieve_chunks

SYSTEM_PROMPT = """You are a technical documentation assistant.
Answer the user's question using ONLY the provided context chunks.
If the context does not contain enough information to answer, say so clearly.
Always cite the source filename for your claims."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c.source} | Chunk {c.chunk_index}]\n{c.content}" for c in chunks
    )


def answer_question(query: str, top_k: int = 5) -> dict:
    """Run the full RAG pipeline: retrieve relevant chunks and generate a grounded answer."""
    chunks = retrieve_chunks(query, top_k=top_k)
    context = format_context(chunks)

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    response = client.chat.completions.create(
        model=settings.chat_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "contexts": [c.content for c in chunks],
        "sources": [
            {"source": c.source, "chunk_index": c.chunk_index, "similarity": c.similarity}
            for c in chunks
        ],
    }


if __name__ == "__main__":
    question = "How do path parameters work in FastAPI?"
    result = answer_question(question)
    print("\n=== ANSWER ===\n")
    print(result["answer"])
    print("\n=== RETRIEVED SOURCES ===")
    for s in result["sources"]:
        print(f"- {s['source']} (chunk {s['chunk_index']}, similarity: {s['similarity']})")