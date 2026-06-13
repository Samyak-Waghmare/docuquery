# DocuQuery — Roadmap

A retrieval-augmented generation (RAG) app that turns any PDF into a cited Q&A
assistant. This roadmap tracks planned improvements, prioritized by impact.

**Current state (MVP, done):** upload → chunk → embed (Gemini) → Qdrant vector
store → semantic retrieval → streaming, page-cited answers. Polished UI with a
live indexing pipeline and "thinking" phases.

---

## Phase 1 — Quick wins (high impact, low effort)

- [ ] **View sources** — expander under each answer showing the actual retrieved
      chunk text + page (not just citation chips). Core trust feature.
- [ ] **Demo document** — "Try with a sample PDF" button (uses `nodejs.pdf`) so a
      reviewer reaches the chat in one click.
- [ ] **Answer actions** — copy, regenerate, 👍/👎 feedback.
- [ ] **Export conversation** — download the chat as Markdown.

## Phase 2 — RAG quality (interview centerpiece)

- [ ] **Hybrid search** — combine dense vectors with BM25/sparse (Qdrant sparse
      vectors) for exact-term queries.
- [ ] **Re-ranking** — cross-encoder or LLM rerank of top-k → top-n.
- [ ] **Evaluation harness** — small Q/A set; measure retrieval hit-rate and
      answer faithfulness (LLM-as-judge). Produce numbers.
- [ ] **MMR / diversity** retrieval to reduce redundant context.

## Phase 3 — Persistence & multi-document

- [ ] **Document library** — list existing Qdrant collections; reopen a doc
      without re-indexing.
- [ ] **Multi-PDF** — query across multiple documents.
- [ ] **Page preview / jump-to-page** beside a citation.

## Phase 4 — Production rigor

- [ ] **README** with architecture diagram, screenshots, and a demo GIF.
- [ ] **App Dockerfile** + unified `docker compose` (app + Qdrant).
- [ ] **Tests** (pytest) + **GitHub Actions CI**.
- [ ] **Observability** — per-query latency, token counts, estimated cost.
- [ ] **Config panel** — model, temperature, chunk size.

## Phase 5 — Stretch / breadth

- [ ] Multi-format ingestion (DOCX, TXT, MD, paste a URL).
- [ ] OCR fallback for scanned PDFs.
- [ ] Long-chat memory summarization.
- [ ] Light/dark theme toggle.
- [ ] Auth + per-user document isolation.

---

### Recommended sequence
Phase 1 → Phase 2 → Phase 4 (README/Docker) → Phase 3.
The combination of **measured retrieval improvement (Phase 2)** + **source
transparency (Phase 1)** + a **strong README** is what makes this stand out.
