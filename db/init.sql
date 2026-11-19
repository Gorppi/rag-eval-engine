CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON document_chunks USING hnsw (embedding vector_cosine_ops);