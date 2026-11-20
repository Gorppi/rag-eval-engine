# rag-eval-engine

A local-first **Retrieval-Augmented Generation (RAG)** backend and automated **LLM-as-a-Judge evaluation harness** built with Python, FastAPI, PostgreSQL (`pgvector`), Ollama, and DeepEval.

Designed to run 100% locally with zero cloud API costs while maintaining a clean, provider-agnostic architecture via the OpenAI-compatible SDK.

## Architecture

The system is organized into three decoupled pipelines:

1. **Offline Ingestion (`src/ingestion/`)** — Parses Markdown documentation, splits content into overlapping chunks, generates dense vector embeddings via local Ollama (`qwen3-embedding:0.6b`), and stores text + vectors in PostgreSQL (`pgvector`).
2. **Online RAG & API (`src/retrieval/`, `src/generation/`, `src/api/`)** — Embeds incoming queries, performs cosine similarity search (`<=>`) in `pgvector` to retrieve top-$k$ chunks, augments a grounded system prompt, and serves cited answers via a typed FastAPI endpoint (`POST /query`).
3. **Automated Evaluation (`evals/`)** — Runs a `pytest`-compatible DeepEval suite against a golden dataset, using a local LLM judge (`OllamaModel`) to score **Faithfulness** (hallucination detection) and **Answer Relevancy**.

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Package & Env Management** | `uv` (`pyproject.toml` + `uv.lock`) |
| **API Framework** | FastAPI + Uvicorn + Pydantic |
| **Vector Database** | PostgreSQL 16 + `pgvector` (Docker Compose) |
| **Local LLM & Embeddings** | Ollama (`qwen3.5:9b`, `qwen3-embedding:0.6b`) |
| **LLM Client** | `openai` Python SDK (pointed at Ollama `/v1`) |
| **Evaluation** | DeepEval + `pytest` |

## Project Structure

```text
rag-eval-engine/
├── db/
│   └── init.sql             # Enables pgvector extension and creates schema
├── evals/
│   ├── dataset.json         # Golden Q&A evaluation dataset
│   └── test_rag.py          # DeepEval Faithfulness & Relevancy test suite
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app and POST /query endpoint
│   │   └── schemas.py       # Pydantic request/response models
│   ├── generation/
│   │   └── rag.py           # Prompt augmentation and LLM generation
│   ├── ingestion/
│   │   ├── chunker.py       # Overlapping text chunker
│   │   └── ingest.py        # Document loader, embedder, and DB writer
│   ├── retrieval/
│   │   └── search.py        # pgvector cosine similarity search
│   └── config.py            # Typed environment settings (pydantic-settings)
├── docker-compose.yml       # Local PostgreSQL + pgvector container (port 5433)
├── .env.example             # Environment variable template
└── pyproject.toml           # Dependencies and pytest configuration
```

## Quick Start

### Prerequisites
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- Docker Desktop
- [Ollama](https://ollama.com/) running locally

### 1. Pull Local Models
```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:9b
```

### 2. Clone & Configure
```bash
git clone [https://github.com/Gorppi/rag-eval-engine.git](https://github.com/Gorppi/rag-eval-engine.git)
cd rag-eval-engine
cp .env.example .env
uv sync
```

### 3. Start the Vector Database
```bash
docker compose up -d
```
*(Mapped to localhost port `5433` to avoid conflicts with local PostgreSQL installations.)*

### 4. Ingest Documents
Place your Markdown documentation in `data/` and run:
```bash
uv run python -m src.ingestion.ingest
```

### 5. Run the API
```bash
uv run uvicorn src.api.main:app --reload
```
Open **`http://127.0.0.1:8000/docs`** to test `POST /query` interactively via Swagger UI.

### 6. Run DeepEval Benchmarks
```bash
uv run deepeval test run evals/test_rag.py
```

## Evaluation Benchmarks

| Pipeline Version | Retrieval Method | Faithfulness | Answer Relevancy |
| :--- | :--- | :---: | :---: |
| **v0.1 (Baseline)** | Dense Vector Search (`pgvector`) | *Pending* | *Pending* |
| **v0.2 (Planned)** | Hybrid Search (`pgvector` + `tsvector` + RRF) | *Pending* | *Pending* |

## Roadmap

- [x] Dockerized PostgreSQL + `pgvector` setup
- [x] End-to-end local ingestion, retrieval, and FastAPI query endpoint
- [x] Automated DeepEval evaluation harness with local Ollama judge
- [ ] Full documentation ingestion & expanded golden dataset
- [ ] Hybrid search combining dense vectors with PostgreSQL full-text search (`tsvector` + Reciprocal Rank Fusion)
- [ ] Decoupled lightweight judge model configuration for faster evaluation cyclesgit add README.md