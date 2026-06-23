# Horizon Tours RAG API

A production-style Retrieval-Augmented Generation (RAG) service that answers questions over the **Horizon Tours & Travel** knowledge base, built with **FastAPI**, **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Groq**.

---

## Project Overview

This service implements a classic RAG architecture:

1. A PDF knowledge base is **indexed offline** — loaded, cleaned, chunked, embedded, and stored in a local ChromaDB vector store.
2. A **FastAPI service** loads that pre-built vector store at startup and answers natural-language questions by retrieving relevant chunks and passing them to a Groq-hosted LLM as grounded context.

The two workflows are deliberately independent:

- **Indexing** (`scripts/build_index.py`) never calls an LLM — it only builds embeddings and writes to disk.
- **Serving** (`app/main.py`) never rebuilds embeddings — it only loads the existing vector store and answers questions.

This separation keeps indexing cheap to re-run and the API fast to start, and mirrors how RAG systems are typically deployed in production (a batch/offline ingestion job feeding a stateless, horizontally-scalable query service).

### Architecture Diagram

```text
PDF
 │
 ▼
Loader
 │
 ▼
Cleaner
 │
 ▼
Chunker
 │
 ▼
Embeddings
 │
 ▼
ChromaDB
 │
 ▼
Retriever
 │
 ▼
Groq LLM
 │
 ▼
Answer
```

---

## Project Structure

```text
rag-fastapi/
├── .github/workflows/test.yml   # CI: lint, test, import check
├── app/
│   ├── main.py                  # FastAPI app, lifespan startup validation
│   ├── api/routes.py            # Thin HTTP routes (no business logic)
│   ├── config/settings.py       # Centralized env-based configuration
│   ├── ingestion/               # PDF loading, cleaning, chunking
│   ├── embeddings/              # HuggingFace embedding model factory
│   ├── vectorstore/             # Chroma build/load (strictly separated)
│   ├── retrieval/               # Retriever + source formatting
│   ├── llm/                     # Groq client factory
│   ├── chains/                  # RAG chain composition
│   ├── prompts/                 # RAG system/user prompt templates
│   ├── schemas/                 # Pydantic request/response models
│   ├── services/rag_service.py  # All business logic lives here
│   └── utils/logger.py          # Centralized logging
├── scripts/build_index.py       # Indexing workflow entry point
├── tests/                       # test_health.py, test_ask.py
├── data/{pdfs,chroma}/          # Source PDFs and persisted vector store
├── requirements.txt
├── pytest.ini
├── ruff.toml
├── Makefile
├── .env.example
└── .gitignore
```

---

## Installation

### Create a virtual environment

```bash
cd RAG-FASTAPI
```

```bash
python -m venv .venv
```

### Activate it

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/Mac**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Copy `.env.example` to `.env` and fill in your Groq API key:

```bash
cp .env.example .env
```

---

## Workflow 1 — Build the Vector Database

Run this once (and again whenever the PDF changes):

```bash
python scripts/build_index.py
```

This loads the PDF from `PDF_PATH`, cleans and chunks the text, generates embeddings, and persists them to `data/chroma/`. **No LLM calls occur during this step.**

---

## Workflow 2 — Run the API

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

On startup, the app validates that `GROQ_API_KEY` is set and that a Chroma collection already exists — it will refuse to start otherwise. It then loads the embedding model, vector store, retriever, and RAG chain **once**, storing them in `app.state` for reuse across all requests.

### Swagger UI

```text
http://localhost:8000/docs
```

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "vectorstore_loaded": true,
  "model": "llama-3.3-70b-versatile"
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the emergency travel assistance number?"}'
```

```json
{
  "question": "What is the emergency travel assistance number?",
  "answer": "...",
  "sources": [
    { "page": 5, "section": "Emergency Support" }
  ]
}
```

---

## Running Tests & the Linter

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m ruff check .
```

`test_ask.py` fully mocks the RAG service — no real calls to Groq or Chroma are made during tests.

---

## Continuous Integration

`.github/workflows/test.yml` runs on every push to `main`/`develop` and on every pull request:

1. Checks out the code
2. Sets up Python 3.11
3. Installs dependencies
4. Runs **Ruff** for linting
5. Runs **Pytest** for the test suite
6. Verifies the application imports cleanly (`from app.main import app`)

The workflow fails on any error, so broken code or lint violations cannot be merged.

---

## Configuration Reference

All configuration is environment-driven (see `.env.example`):

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key | — |
| `GROQ_MODEL` | Groq model identifier | `llama-3.3-70b-versatile` |
| `PDF_PATH` | Path to the source PDF | `data/pdfs/Horizon_Tours_Complete_Knowledge_Base_2025.pdf` |
| `CHROMA_DB_DIR` | Chroma persistence directory | `data/chroma` |
| `COLLECTION_NAME` | Chroma collection name | `horizon_tours_kb` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `BAAI/bge-small-en-v1.5` |
| `TOP_K` | Chunks retrieved per query | `5` |
| `FETCH_K` | Candidates considered before MMR selection | `12` |
| `CHUNK_SIZE` | Max characters per chunk | `1200` |
| `CHUNK_OVERLAP` | Overlap characters between chunks | `200` |
| `LLM_TEMPERATURE` | Groq sampling temperature | `0.1` |
| `LLM_MAX_TOKENS` | Max tokens per Groq response | `600` |
| `API_HOST` | FastAPI host | `0.0.0.0` |
| `API_PORT` | FastAPI port | `8000` |
| `LOG_LEVEL` | Python logging level | `INFO` |

---

## Code Quality

This project follows:

- **Type hints** throughout
- **Docstrings** on every public function/class
- **Dependency injection** via `Settings` and `app.state`
- **Structured logging** at every pipeline stage
- **Pydantic validation** on all API I/O
- **Clean architecture**: ingestion / embeddings / vectorstore / retrieval / llm / chains layers are independently swappable
- **SOLID principles** and strict separation of concerns (routes contain no business logic; that lives in `services/`)

---
