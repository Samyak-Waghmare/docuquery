1"""
DocuQuery — RAG Evaluation Script
===================================
Measures Recall@K and MRR (Mean Reciprocal Rank) for the retrieval pipeline.

Usage:
    python evaluate_rag.py [--collection <name>] [--k <int>]

Requirements:
    - A running Qdrant instance (local or cloud).
    - eval_set.json in the same directory.
    - GEMINI_API_KEY and QDRANT_URL set in .env
"""

import os, sys, json, time, argparse
from pathlib import Path
from dotenv import load_dotenv
import io

# Force UTF-8 stdout to avoid Windows cp1252 errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
QDRANT_URL      = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY") or None
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "docuquery")
EVAL_FILE       = Path(__file__).parent / "eval_set.json"

if not GEMINI_API_KEY:
    sys.stdout.write("[ERR] GEMINI_API_KEY not set. Add it to your .env file.\n")
    sys.exit(1)

# ── Argument Parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DocuQuery RAG Evaluation")
parser.add_argument("--collection", default=COLLECTION_NAME,
                    help="Qdrant collection name (default: from .env)")
parser.add_argument("--k", type=int, default=5,
                    help="Top-K results to retrieve (default: 5)")
args = parser.parse_args()

COLLECTION_NAME = args.collection
TOP_K           = args.k

# ── Lazy Imports ──────────────────────────────────────────────────────────────
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
except ImportError:
    print("[ERR] Missing dependencies. Run: pip install langchain-google-genai langchain-qdrant")
    sys.exit(1)

# ── Load eval set ─────────────────────────────────────────────────────────────
if not EVAL_FILE.exists():
    print(f"[ERR] eval_set.json not found at {EVAL_FILE}")
    sys.exit(1)

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_set = json.load(f)

print(f"\n{'='*60}")
print(f"  DocuQuery RAG Evaluation Framework")
print(f"{'='*60}")
print(f"  Collection : {COLLECTION_NAME}")
print(f"  Top-K      : {TOP_K}")
print(f"  Test Items : {len(eval_set)}")
print(f"{'='*60}\n")

# ── Connect to Qdrant ─────────────────────────────────────────────────────────
print("[*] Connecting to Qdrant vector store...")
try:
    emb = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )
    vs = QdrantVectorStore.from_existing_collection(
        embedding=emb,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )
    print(f"[OK] Connected to collection '{COLLECTION_NAME}'\n")
except Exception as e:
    print(f"[ERR] Failed to connect to Qdrant: {e}")
    print("   Make sure Qdrant is running and you have indexed a document first.")
    sys.exit(1)

# ── Run Evaluation ────────────────────────────────────────────────────────────
results = []
latencies = []

print(f"{'─'*60}")
print(f"  Running {len(eval_set)} queries…")
print(f"{'─'*60}")

for i, item in enumerate(eval_set, 1):
    q           = item["q"]
    answer_page = item.get("answer_page")
    must_contain = item.get("must_contain", [])

    t0 = time.time()
    try:
        docs = vs.similarity_search(query=q, k=TOP_K)
        latency = time.time() - t0
    except Exception as e:
        print(f"  [{i:02d}] ⚠️  Search failed: {e}")
        latency = None
        docs = []

    if latency is not None:
        latencies.append(latency)

    # Extract page numbers from metadata
    retrieved_pages = []
    for doc in docs:
        page = doc.metadata.get("page") or doc.metadata.get("page_number")
        if page is not None:
            retrieved_pages.append(int(page))

    # Recall: did the correct page appear?
    hit = (answer_page in retrieved_pages) if answer_page is not None else None

    # Reciprocal Rank: position of first correct page
    rr = 0.0
    if answer_page is not None:
        for rank, page in enumerate(retrieved_pages, 1):
            if page == answer_page:
                rr = 1.0 / rank
                break

    results.append({"q": q, "hit": hit, "rr": rr, "pages": retrieved_pages, "latency": latency})

    status = "[HIT]" if hit else ("[???]" if hit is None else "[MISS]")
    lat_str = f"{latency:.2f}s" if latency else "n/a"
    print(f"  [{i:02d}] {status}  Pages retrieved: {retrieved_pages}  |  Expected: p.{answer_page}  |  {lat_str}")

# ── Compute Metrics ───────────────────────────────────────────────────────────
valid_hits   = [r for r in results if r["hit"] is not None]
recall_at_k  = sum(r["hit"] for r in valid_hits) / len(valid_hits) if valid_hits else 0
mrr          = sum(r["rr"]  for r in valid_hits) / len(valid_hits) if valid_hits else 0
p50_latency  = sorted(latencies)[len(latencies) // 2] if latencies else 0
p95_latency  = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
avg_latency  = sum(latencies) / len(latencies) if latencies else 0

# ── Final Report ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  EVALUATION RESULTS")
print(f"{'='*60}")
print(f"  Recall@{TOP_K}    : {recall_at_k:.3f}  ({sum(r['hit'] for r in valid_hits if r['hit'])}/{len(valid_hits)} correct pages found)")
print(f"  MRR          : {mrr:.3f}  (closer to 1.0 is better)")
print(f"{'─'*60}")
print(f"  Avg Latency  : {avg_latency:.2f}s")
print(f"  p50 Latency  : {p50_latency:.2f}s")
print(f"  p95 Latency  : {p95_latency:.2f}s")
print(f"{'─'*60}")

# Interpretation
print(f"\n  INTERPRETATION:")
if recall_at_k >= 0.85:
    print(f"  [EXCELLENT] Recall@{TOP_K} >= 0.85")
elif recall_at_k >= 0.70:
    print(f"  [GOOD] Recall@{TOP_K} >= 0.70, but room to improve.")
    print(f"     Consider: Larger chunk overlap, or Hybrid Search (BM25 + dense).")
else:
    print(f"  [POOR] Recall@{TOP_K} < 0.70 - retrieval needs work.")
    print(f"     Consider: Hybrid Search, smaller chunk size, or a reranker.")

if mrr < 0.5:
    print(f"  [WARN] MRR is low - the correct page is often not ranked #1.")
    print(f"     A reranker (e.g., cross-encoder) can push the right chunk higher.")

print(f"\n  SAVE: Full results saved to eval_results.json")

# ── Save results to JSON ──────────────────────────────────────────────────────
out_file = Path(__file__).parent / "eval_results.json"
with open(out_file, "w") as f:
    json.dump({
        "collection": COLLECTION_NAME,
        "top_k": TOP_K,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "avg_latency": avg_latency,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "results": results,
    }, f, indent=2)

print(f"📄 Full results saved to eval_results.json")
