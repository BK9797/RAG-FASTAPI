"""Indexing workflow entry point.

Run this script whenever a new PDF needs to be indexed:

    python scripts/build_index.py

This script ONLY builds embeddings and persists them to ChromaDB.
It makes NO calls to the Groq LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly (python scripts/build_index.py)
# by ensuring the project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings  # noqa: E402
from app.embeddings.embedding_model import get_embedding_model  # noqa: E402
from app.ingestion.chunker import chunk_pages  # noqa: E402
from app.ingestion.pdf_loader import load_pdf_pages  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from app.vectorstore.chroma_store import build_vectorstore  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Run the full indexing pipeline: load -> clean -> chunk -> embed -> store."""
    logger.info("Loading settings...")
    settings = get_settings()

    pdf_path = Path(settings.pdf_path)
    if not pdf_path.exists():
        logger.error("PDF not found at: %s", pdf_path)
        raise FileNotFoundError(
            f"PDF not found at: {pdf_path}. Set PDF_PATH in .env or place the file there."
        )

    logger.info("Starting indexing workflow for: %s", pdf_path)

    # 1-2) Load PDF and extract text (cleaning happens inside load_pdf_pages)
    pages = load_pdf_pages(pdf_path)

    # 3-4) Clean (already applied) and chunk text
    documents = chunk_pages(
        pages=pages,
        source_name=pdf_path.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not documents:
        logger.error("No text could be extracted from the PDF. Aborting.")
        raise ValueError("No chunks were produced from the PDF; nothing to index.")

    # 5) Generate embeddings (model only; no LLM calls)
    embeddings = get_embedding_model(settings.embedding_model)

    # 6) Store embeddings in ChromaDB
    build_vectorstore(
        documents=documents,
        embeddings=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_db_dir,
    )

    logger.info(
        "Indexing complete. %d chunks stored in collection '%s' at '%s'.",
        len(documents),
        settings.collection_name,
        settings.chroma_db_dir,
    )


if __name__ == "__main__":
    main()
