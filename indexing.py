"""
DocuQuery — Standalone PDF Indexer (CLI)
─────────────────────────────────────────────────────────────────────────────
Loads a PDF, splits it into chunks, embeds them with Gemini, and stores the
vectors in a Qdrant collection. Embedding runs in rate-limited "waves" so it
stays inside the Gemini free-tier limit.

Usage:
    python indexing.py <path/to/file.pdf> [collection_name]

NOTE: This file was reconstructed from fragments after an accidental deletion.
Review it against your original intent before relying on it.
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
QDRANT_URL      = os.environ.get("QDRANT_URL",      "http://localhost:6333")
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY") or None

# Embedding rate-limit (free tier) controls
WAVE    = 80
PAUSE_S = 55.0
BATCH   = 80


def index_pdf(pdf_path: str, collection: str = "learning_vectors") -> dict:
    """Index a single PDF into a Qdrant collection. Returns a stats dict."""
    # 1. Load the PDF
    print(f"\n📂 Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs   = loader.load()
    print(f"   ✅ {len(docs)} pages loaded")

    # 2. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"   ✅ {len(chunks)} chunks created")

    # 3. Embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

    # 4. Store vectors in Qdrant (rate-limited waves)
    print(f"\n🔢 Indexing into Qdrant collection: '{collection}'")
    print(f"   URL: {QDRANT_URL}")

    first, rest = chunks[:WAVE], chunks[WAVE:]
    vector_store = QdrantVectorStore.from_documents(
        documents=first,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection,
        batch_size=BATCH,
    )
    print(f"   Wave 1/{1 + (len(rest) // WAVE + 1) if rest else 1}: {len(first)} chunks ✅")

    wave_num = 1
    for i in range(0, len(rest), WAVE):
        wave_num += 1
        batch = rest[i:i + WAVE]
        print(f"   Pausing {PAUSE_S}s (rate limit)…", end="", flush=True)
        time.sleep(PAUSE_S)
        vector_store.add_documents(batch, batch_size=BATCH)
        print(f" Wave {wave_num}: {len(batch)} chunks ✅")

    return {"pages": len(docs), "chunks": len(chunks), "collection": collection}


if __name__ == "__main__":
    pdf_path   = sys.argv[1] if len(sys.argv) > 1 else "nodejs.pdf"
    collection = sys.argv[2] if len(sys.argv) > 2 else "learning_vectors"

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    stats = index_pdf(pdf_path, collection)
    print(f"\n🎉 Indexing complete!")
    print(f"   Pages    : {stats['pages']}")
    print(f"   Chunks   : {stats['chunks']}")
    print(f"   Collection: {stats['collection']}")
