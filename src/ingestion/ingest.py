# src/ingestion/ingest.py
from pathlib import Path

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

from src.config import settings
from src.ingestion.chunker import chunk_text

DOCS_DIR = Path("data/fastapi-repo/docs/en/docs")  # Update if you chose different docs
BATCH_SIZE = 32


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of text chunks."""
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_documents(docs_dir: Path = DOCS_DIR, max_files: int | None = 20) -> None:
    """Read Markdown files, chunk them, embed them, and store them in Postgres."""
    md_files = sorted(docs_dir.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {docs_dir.resolve()}")

    # Start with a subset (20 files) so your first run finishes in under a minute
    if max_files is not None:
        md_files = md_files[:max_files]

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        total_chunks = 0

        for file_path in md_files:
            rel_path = str(file_path.relative_to(docs_dir))
            text = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(text)

            if not chunks:
                continue

            # Embed in batches for speed
            embeddings: list[list[float]] = []
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                embeddings.extend(get_embeddings(client, batch))

            with conn.cursor() as cur:
                # Make re-runs idempotent by replacing old chunks for this file
                cur.execute("DELETE FROM document_chunks WHERE source = %s", (rel_path,))
                cur.executemany(
                    """
                    INSERT INTO document_chunks (source, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (rel_path, idx, chunk, emb)
                        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
                    ],
                )
            conn.commit()
            total_chunks += len(chunks)
            print(f"Ingested {rel_path} ({len(chunks)} chunks)")

        print(f"\nDone! Stored {total_chunks} total chunks from {len(md_files)} files.")


if __name__ == "__main__":
    ingest_documents()