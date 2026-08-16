"""
DocuQuery — AI-Powered Document Intelligence
Sidebar-free, production UI: centered upload → chat, robust RAG pipeline.
"""

import os
import time
import tempfile
import logging
from functools import lru_cache
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Full error detail goes to the SERVER log only — never to the browser.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docuquery")


# ─── Favicon: render the DocuQuery logo (document + magnifier + spark) ─────────
@lru_cache(maxsize=1)
def _favicon():
    """Draw the brand mark as a PIL image to use as the browser-tab icon."""
    from PIL import Image, ImageDraw
    S = 256
    # diagonal violet→blue gradient
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    top, bot = (167, 139, 250), (96, 165, 250)
    for y in range(S):
        t = y / (S - 1)
        gd.line([(0, y), (S, y)], fill=(
            int(top[0] + (bot[0] - top[0]) * t),
            int(top[1] + (bot[1] - top[1]) * t),
            int(top[2] + (bot[2] - top[2]) * t), 255))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([6, 6, S - 6, S - 6], radius=58, fill=255)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(img)
    s = S / 24.0
    W = (255, 255, 255, 255)
    # page — OPAQUE fill (translucent fills render black on dark browser tabs)
    d.rounded_rectangle([5 * s, 3 * s, 17 * s, 20.6 * s], radius=2 * s,
                        outline=W, width=max(2, int(0.55 * s)), fill=(255, 255, 255, 255))
    lw = max(2, int(0.5 * s))
    LINE = (150, 139, 210, 255)  # violet-gray "text" lines, visible on the white page
    d.line([(6.6 * s, 11 * s), (11 * s, 11 * s)], fill=LINE, width=lw)
    d.line([(6.6 * s, 13.4 * s), (9.3 * s, 13.4 * s)], fill=LINE, width=lw)
    # magnifier lens (drawn over the page)
    cx, cy, r = 15 * s, 14.8 * s, 3.7 * s
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(12, 10, 24, 255),
              outline=W, width=max(2, int(0.6 * s)))
    # 4-point spark inside the lens
    sp = 1.6 * s
    d.polygon([(cx, cy - sp), (cx + 0.36 * sp, cy - 0.36 * sp), (cx + sp, cy),
               (cx + 0.36 * sp, cy + 0.36 * sp), (cx, cy + sp),
               (cx - 0.36 * sp, cy + 0.36 * sp), (cx - sp, cy),
               (cx - 0.36 * sp, cy - 0.36 * sp)], fill=W)
    # handle
    d.line([(17.8 * s, 17.6 * s), (20 * s, 19.8 * s)], fill=W, width=max(2, int(0.7 * s)))
    return img


# ─── Page Config — MUST be first ───────────────────────────────────────────────
st.set_page_config(
    page_title="DocuQuery — AI Document Intelligence",
    page_icon=_favicon(),
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Load CSS from file ─────────────────────────────────────────────────────────
def load_css():
    p = Path(__file__).parent / "style.css"
    if p.exists():
        st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)
load_css()

# ─── Env / secrets ─────────────────────────────────────────────────────────────
load_dotenv()

def _secret(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    if v:
        return v
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

GEMINI_API_KEY  = _secret("GEMINI_API_KEY")
QDRANT_URL      = _secret("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = _secret("QDRANT_API_KEY") or None
COLLECTION_BASE = _secret("QDRANT_COLLECTION", "docuquery")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ─── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "messages":        [],
    "vector_store":    None,
    "indexed_file":    None,
    "total_chunks":    0,
    "page_count":      0,
    "status":          "idle",   # idle | indexing | ready | error
    "error_msg":       "",
    "collection_name": "",
    "pending_prompt":  None,
    "top_k":           4,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Lazy imports (avoid loading heavy libs at startup) ────────────────────────
@st.cache_resource(show_spinner=False)
def _emb_model():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

@st.cache_resource(show_spinner=False)
def _llm():
    from openai import OpenAI
    return OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

# ─── Indexing ──────────────────────────────────────────────────────────────────
def do_index(file_bytes: bytes, filename: str, on_step=None):
    """Full RAG indexing pipeline. Returns (vector_store | False, error_string).

    `on_step(stage, message, state)` is called before each phase so the UI can
    show live progress. `state` is 'run' | 'done'.
    """
    import tempfile, time, re
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    def step(stage, message, state="run"):
        if on_step:
            on_step(stage, message, state)

    try:
        # 1. Receive the upload
        step("upload", f"Uploaded **{filename}** · {len(file_bytes)/1024:.0f} KB")
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(file_bytes)
        tmp.close()

        # 2. Parse / extract text
        step("parse", "Reading and extracting text from the PDF…")
        loader = PyPDFLoader(tmp.name)
        docs   = loader.load()
        if not docs:
            return False, "PDF appears to be empty or unreadable."
        st.session_state.page_count = len(docs)
        step("parse", f"Extracted text from **{len(docs)}** pages", "done")

        # 3. Chunk
        step("chunk", "Splitting text into overlapping chunks…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        st.session_state.total_chunks = len(chunks)
        step("chunk", f"Created **{len(chunks)}** chunks (1000 chars, 150 overlap)", "done")

        # 4. Build a safe collection name
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", filename[:30])
        coll = f"{COLLECTION_BASE}_{safe}"
        st.session_state.collection_name = coll

        # 5. Connect to vector DB + clean re-index
        step("store", "Connecting to the Qdrant vector database…")
        try:
            qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            existing = [c.name for c in qc.get_collections().collections]
            if coll in existing:
                qc.delete_collection(coll)
        except Exception:
            pass  # non-fatal

        # 6. Embed + store with robust retry logic (handles Gemini free-tier rate limits)
        emb  = _emb_model()
        BATCH_SIZE = 20
        total_batches = max(1, (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE)

        vs = None
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            step("embed", f"Embedding batch {batch_num}/{total_batches}…")
            
            for attempt in range(5):
                try:
                    if vs is None:
                        # First batch creates the Qdrant collection
                        vs = QdrantVectorStore.from_documents(
                            documents=batch,
                            embedding=emb,
                            url=QDRANT_URL,
                            api_key=QDRANT_API_KEY,
                            collection_name=coll,
                        )
                    else:
                        # Subsequent batches add to the collection
                        vs.add_documents(batch)
                    break  # Success, exit retry loop
                except Exception as e:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "rate" in err_msg or "quota" in err_msg or "exhausted" in err_msg:
                        wait_time = 2 ** attempt * 8  # 8s, 16s, 32s, 64s, 128s
                        step("embed", f"API rate limit hit. Pausing {wait_time}s to recover (batch {batch_num})…")
                        time.sleep(wait_time)
                    else:
                        raise e
            else:
                 raise RuntimeError("Gemini API rate limit exceeded even after 5 retries. Try a smaller PDF.")
            
            # Small standard delay between batches to avoid hitting limits
            if i + BATCH_SIZE < len(chunks):
                time.sleep(3)

        step("embed", f"Embedded & stored **{len(chunks)}** vectors", "done")
        step("ready", "Index ready — you can start asking questions", "done")

        # 7. Cleanup
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

        return vs, None

    except Exception as e:
        # Log the full traceback to the SERVER only; return a short, safe label.
        logger.exception("Indexing failed")
        return False, type(e).__name__ + ": " + str(e)

def friendly_error(raw: str) -> str:
    """Map an error to a safe, user-facing hint — never expose paths/URLs/keys."""
    low = (raw or "").lower()
    if "10061" in raw or "actively refused" in low or ("connection" in low and "refused" in low) \
       or "timed out" in low or "name or service" in low or "getaddrinfo" in low:
        return ("Can't reach the vector database. Check that the vector store is "
                "running and that its connection settings are configured correctly.")
    if "api key" in low or "api_key" in low or "permission" in low or "401" in raw or "403" in raw \
       or "unauthorized" in low:
        return "The AI service rejected the request — its API credentials may be missing or invalid."
    if "resource_exhausted" in low or "429" in raw or "quota" in low or "rate" in low:
        return "The AI service is rate-limited right now. Please wait a minute and try again."
    # Generic fallback — do NOT echo raw internals to the browser.
    return "Something went wrong while processing the document. Please try again."
    return raw.split("\n", 1)[0]

# ─── RAG helpers ───────────────────────────────────────────────────────────────
def build_context(results):
    parts, cits = [], []
    seen_pages = set()
    for i, r in enumerate(results, 1):
        page = r.metadata.get("page_label", r.metadata.get("page", "?"))
        src  = Path(r.metadata.get("source", "PDF")).name
        parts.append(f"[Source {i}] Page {page} — {src}\n{r.page_content}")
        if str(page) not in seen_pages:
            cits.append({"index": i, "page": page})
            seen_pages.add(str(page))
    context_str = "\n\n---\n\n".join(parts)
    if len(context_str) > 6000:
        context_str = context_str[:6000] + "\n\n[...context trimmed for length...]"
    return context_str, cits

def history_msgs(window=6):
    return [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-window:]]

def stream_answer(query, context, history):
    sys_prompt = f"""You are DocuQuery, an expert AI assistant that answers
questions strictly from the provided PDF document context.

Rules:
- Answer ONLY using the provided context. If the answer isn't there, say so clearly.
- Always cite [Source N] and the page number when using information.
- Give a COMPLETE, detailed answer — do not stop mid-sentence.
- Use bullet points or numbered lists for clarity when helpful.
- After your answer, mention the page numbers the user can refer to.

DOCUMENT CONTEXT:
{context}"""
    msgs = [{"role": "system", "content": sys_prompt}] + history + \
           [{"role": "user",   "content": query}]
    stream = _llm().chat.completions.create(
        model="gemini-2.5-flash",
        messages=msgs,
        stream=True,
        temperature=0.2,
        max_tokens=2048,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

def cite_chips(citations) -> str:
    return " ".join(f'<span class="dq-cite">📄 p.{c["page"]}</span>' for c in citations)

def reset_to_upload():
    st.session_state.status        = "idle"
    st.session_state.vector_store  = None
    st.session_state.indexed_file  = None
    st.session_state.messages      = []
    st.session_state.error_msg     = ""

# ══════════════════════════════════════════════════════════════════════════════
# HEADER (brand bar) — always visible
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dq-topbar">
  <div class="dq-logo">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M5.4 2.6H12l5.1 5.1v9.4a2 2 0 0 1-2 2H5.4a2 2 0 0 1-2-2V4.6a2 2 0 0 1 2-2Z"
            fill="rgba(255,255,255,0.2)" stroke="white" stroke-width="1.3" stroke-linejoin="round"/>
      <path d="M11.7 2.8V7a1 1 0 0 0 1 1h4.1" stroke="white" stroke-width="1.3"
            stroke-linejoin="round" stroke-linecap="round"/>
      <path d="M6.5 11h4.4M6.5 13.4h2.7" stroke="rgba(255,255,255,0.85)" stroke-width="1.2" stroke-linecap="round"/>
      <circle cx="15" cy="14.8" r="3.7" fill="rgba(6,5,9,0.9)" stroke="white" stroke-width="1.4"/>
      <path d="M15 12.9l.55 1.35 1.35.55-1.35.55L15 16.7l-.55-1.35-1.35-.55 1.35-.55L15 12.9Z" fill="white"/>
      <path d="M17.8 17.6 19.9 19.7" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
    </svg>
  </div>
  <div>
    <div class="dq-brand-name">DocuQuery</div>
    <div class="dq-brand-sub">AI Document Intelligence</div>
  </div>
  <div class="dq-badges">
    <span class="dq-badge green">✦ RAG Pipeline</span>
    <span class="dq-badge violet">📍 Page Citations</span>
    <span class="dq-badge blue">⚡ Gemini 2.5</span>
  </div>
</div>
<div style="height:20px"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STATE: IDLE  → centered hero + upload dropzone
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.status in ("idle", "indexing"):

    if st.session_state.status == "idle":
        st.markdown("""
        <div class="dq-hero">
          <div class="dq-hero-badge">✦ Powered by Gemini 2.5 &amp; Qdrant</div>
          <div class="dq-hero-icon">📑</div>
          <h1>Chat with <span class="grad">any PDF,</span><br>get cited answers</h1>
          <p>Drop in a document — research paper, contract, manual, report — and
             instantly query it with AI. Every answer comes with exact page references
             so you can verify what matters.</p>
        </div>
        """, unsafe_allow_html=True)

        if not GEMINI_API_KEY:
            st.markdown("""
            <div class="dq-keywarn">
              <span>⚠️</span>
              <span><b>No API key found.</b> Set <code>GEMINI_API_KEY</code> in your
              <code>.env</code> file to enable indexing and chat.</span>
            </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload PDF", type=["pdf"], label_visibility="collapsed",
            help="Upload any PDF (max 200 MB). It's chunked and indexed automatically.",
        )

        st.markdown("""
        <div class="dq-section" style="margin-top:68px;">
          <div class="dq-eyebrow">How it works</div>
          <h2 class="dq-section-title">From PDF to answers in three steps</h2>
        </div>
        <div class="dq-cards">
          <div class="dq-card">
            <div class="ic" style="background:linear-gradient(135deg,rgba(167,139,250,0.2),rgba(167,139,250,0.08));color:#a78bfa;font-size:1.3rem;">①</div>
            <div class="t">Upload your PDF</div>
            <div class="d">Drag &amp; drop or browse — research papers, legal contracts, technical manuals, financial reports.</div>
          </div>
          <div class="dq-card">
            <div class="ic" style="background:linear-gradient(135deg,rgba(129,140,248,0.2),rgba(129,140,248,0.08));color:#818cf8;font-size:1.3rem;">②</div>
            <div class="t">Auto-indexed via RAG</div>
            <div class="d">Text is parsed, chunked into semantic segments, and embedded into a high-speed vector database.</div>
          </div>
          <div class="dq-card">
            <div class="ic" style="background:linear-gradient(135deg,rgba(52,211,153,0.2),rgba(52,211,153,0.08));color:#34d399;font-size:1.3rem;">③</div>
            <div class="t">Ask anything</div>
            <div class="d">Get grounded, cited answers with exact page references — powered by Gemini 2.5 Flash.</div>
          </div>
        </div>

        <div class="dq-section" style="margin-top:100px;">
          <div class="dq-eyebrow">Under the hood</div>
          <h2 class="dq-section-title">A production-grade RAG pipeline</h2>
          <p class="dq-section-sub">Every answer is grounded in your document — retrieved by
             semantic vector search, then synthesized by Gemini and cited back to the source.
             Zero hallucinations on topics outside the document.</p>
        </div>
        <div class="dq-pipeline">
          <div class="dq-pipe"><span class="ico">📄</span><span class="lb">Upload</span></div>
          <div class="dq-arrow">→</div>
          <div class="dq-pipe"><span class="ico">✂️</span><span class="lb">Chunk</span></div>
          <div class="dq-arrow">→</div>
          <div class="dq-pipe"><span class="ico">🔢</span><span class="lb">Embed</span></div>
          <div class="dq-arrow">→</div>
          <div class="dq-pipe"><span class="ico">🗃️</span><span class="lb">Qdrant</span></div>
          <div class="dq-arrow">→</div>
          <div class="dq-pipe"><span class="ico">🎯</span><span class="lb">Retrieve</span></div>
          <div class="dq-arrow">→</div>
          <div class="dq-pipe accent"><span class="ico">✦</span><span class="lb">Cited&nbsp;Answer</span></div>
        </div>

        <div class="dq-section" style="margin-top:100px;">
          <div class="dq-eyebrow">Capabilities</div>
          <h2 class="dq-section-title">Built to be accurate and verifiable</h2>
        </div>
        <div class="dq-features">
          <div class="dq-feature">
            <div class="fic">📍</div>
            <div class="ft">Page-level citations</div>
            <div class="fd">Every answer links back to the exact page number — verify claims in seconds.</div>
          </div>
          <div class="dq-feature">
            <div class="fic">🔎</div>
            <div class="ft">Semantic retrieval</div>
            <div class="fd">Understands meaning, not just keywords — vector similarity over embedded chunks.</div>
          </div>
          <div class="dq-feature">
            <div class="fic">💬</div>
            <div class="ft">Multi-turn memory</div>
            <div class="fd">Follow-up questions build on prior context — like a real conversation.</div>
          </div>
          <div class="dq-feature">
            <div class="fic">⚡</div>
            <div class="ft">Streaming answers</div>
            <div class="fd">Token-by-token streaming via Gemini 2.5 Flash — fast and responsive.</div>
          </div>
          <div class="dq-feature">
            <div class="fic">🔒</div>
            <div class="ft">Self-hosted vectors</div>
            <div class="fd">Your document index lives in your own Qdrant instance — private by default.</div>
          </div>
          <div class="dq-feature">
            <div class="fic">🎛️</div>
            <div class="ft">Tunable retrieval</div>
            <div class="fd">Adjust Top-K chunks per query to balance broad recall vs. focused precision.</div>
          </div>
        </div>

        <div class="dq-section" style="margin-top:100px;">
          <div class="dq-eyebrow">Built for</div>
          <h2 class="dq-section-title">Any document you need answers from</h2>
        </div>
        <div class="dq-usecases">
          <span class="dq-uc">📚 Research papers</span>
          <span class="dq-uc">⚖️ Legal contracts</span>
          <span class="dq-uc">🛠️ Technical manuals</span>
          <span class="dq-uc">📊 Financial reports</span>
          <span class="dq-uc">📋 Policy documents</span>
          <span class="dq-uc">🎓 Textbooks &amp; study notes</span>
          <span class="dq-uc">🏥 Medical literature</span>
          <span class="dq-uc">📰 Annual reports</span>
        </div>

        <div class="dq-section" style="margin-top:80px;">
          <div class="dq-eyebrow">Tech stack</div>
          <h2 class="dq-section-title">Open-source, production-ready</h2>
        </div>
        <div class="dq-stack">
          <span class="dq-stack-badge">🐍 Python 3.11+</span>
          <span class="dq-stack-badge">🦜 LangChain</span>
          <span class="dq-stack-badge">🗃️ Qdrant Vector DB</span>
          <span class="dq-stack-badge">✦ Gemini 2.5 Flash</span>
          <span class="dq-stack-badge">🔢 Gemini Embeddings</span>
          <span class="dq-stack-badge">🎈 Streamlit</span>
          <span class="dq-stack-badge">🔗 OpenAI-compatible API</span>
        </div>

        <div class="dq-footer">
          <div class="dq-footer-grid">
            <div class="dq-footer-col dq-footer-about">
              <div class="dq-footer-brand">
                <span class="dq-footer-mark">📑</span> DocuQuery
              </div>
              <p>Turn any PDF into an intelligent, citation-backed assistant.
                 Built on a retrieval-augmented generation pipeline — accurate,
                 verifiable, and self-hosted.</p>
            </div>
            <div class="dq-footer-col">
              <h4>Pipeline</h4>
              <span>PDF parsing &amp; extraction</span>
              <span>Semantic text chunking</span>
              <span>Vector embeddings</span>
              <span>Similarity retrieval</span>
              <span>Grounded generation</span>
            </div>
            <div class="dq-footer-col">
              <h4>Built with</h4>
              <span>Python &amp; LangChain</span>
              <span>Qdrant Vector DB</span>
              <span>Google Gemini 2.5</span>
              <span>Streamlit UI</span>
              <span>PyPDF / RecursiveTextSplitter</span>
            </div>
          </div>
          <div class="dq-footer-bar">
            <span>© 2026 DocuQuery — AI Document Intelligence</span>
            <span class="dq-footer-tags">RAG · Vector Search · Page-Cited Answers · Gemini</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Trigger indexing on a new upload
        if uploaded is not None and uploaded.name != st.session_state.indexed_file:
            st.session_state.status = "indexing"
            st.session_state._pending_bytes = uploaded.read()
            st.session_state._pending_name  = uploaded.name
            st.rerun()

    else:  # indexing — live phase-by-phase progress
        name = st.session_state.get("_pending_name", "your document")

        # Visual pipeline tracker that lights up stage-by-stage
        STAGES = [
            ("upload", "📤", "Upload"),
            ("parse",  "📄", "Parse"),
            ("chunk",  "✂️", "Chunk"),
            ("embed",  "🔢", "Embed"),
            ("store",  "🗃️", "Store"),
            ("ready",  "✦",  "Ready"),
        ]
        st.markdown(f"""
        <div class="dq-section" style="margin-top:18px;margin-bottom:6px;">
          <div class="dq-eyebrow">Processing</div>
          <h2 class="dq-section-title">Building the index for {name}</h2>
          <p class="dq-section-sub">Watch your document flow through the RAG pipeline.
             Larger PDFs embed in rate-limited waves — keep this tab open.</p>
        </div>""", unsafe_allow_html=True)

        tracker = st.empty()
        done_stages: set[str] = set()
        active_stage = "upload"

        def render_tracker(active, done):
            html = ['<div class="dq-pipeline dq-track">']
            for j, (key, ico, label) in enumerate(STAGES):
                cls = "dq-pipe"
                if key in done:        cls += " done"
                elif key == active:    cls += " active"
                html.append(f'<div class="{cls}"><span class="ico">{ico}</span>'
                            f'<span class="lb">{label}</span></div>')
                if j < len(STAGES) - 1:
                    html.append('<div class="dq-arrow">→</div>')
            html.append("</div>")
            tracker.markdown("".join(html), unsafe_allow_html=True)

        render_tracker(active_stage, done_stages)

        with st.status("Starting…", expanded=True) as status:
            def on_step(stage, message, state="run"):
                if state == "done":
                    done_stages.add(stage)
                render_tracker(stage, done_stages)
                status.update(label=message.replace("**", ""))
                st.write(("✅ " if state == "done" else "⏳ ") +
                         message.replace("**", ""))

            vs, err = do_index(
                st.session_state.get("_pending_bytes", b""),
                st.session_state.get("_pending_name", "document.pdf"),
                on_step=on_step,
            )
            if err:
                status.update(label="Indexing failed", state="error")
            else:
                status.update(label="Index ready ✓", state="complete", expanded=False)

        if err:
            st.session_state.status    = "error"
            st.session_state.error_msg = err
        else:
            st.session_state.status       = "ready"
            st.session_state.vector_store = vs
            st.session_state.indexed_file = st.session_state.get("_pending_name")
            st.session_state.messages     = []
            st.session_state.error_msg    = ""
        st.session_state.pop("_pending_bytes", None)
        time.sleep(0.5)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STATE: ERROR
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.status == "error":
    # Friendly message only — full diagnostics go to the server log, not the UI.
    st.error(f"**Couldn't process the document.** {friendly_error(st.session_state.error_msg)}")
    if st.button("← Try another document", type="primary"):
        reset_to_upload()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STATE: READY  → chat
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.status == "ready":
    fname = st.session_state.indexed_file or "—"

    # ── Document toolbar: info pills (left) + actions (right) ──────────────────
    info_col, set_col, new_col = st.columns([0.6, 0.21, 0.19])
    with info_col:
        st.markdown(f"""
        <div class="dq-docrow">
          <span class="dq-pill"><span class="dq-dot"></span>{fname}</span>
          <span class="dq-pill">📄 <b>{st.session_state.page_count}</b>&nbsp;pages</span>
          <span class="dq-pill">🧩 <b>{st.session_state.total_chunks}</b>&nbsp;chunks</span>
        </div>""", unsafe_allow_html=True)
    with set_col:
        with st.popover("⚙ Settings", use_container_width=True):
            st.slider("Top-K Retrieval", 2, 8, key="top_k",
                      help="How many document chunks to retrieve per question.")
            if st.button("🗑 Clear conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    with new_col:
        if st.button("＋ New", use_container_width=True, help="Upload a different document"):
            reset_to_upload()
            st.rerun()

    st.markdown('<hr class="dq-divider">', unsafe_allow_html=True)

    # ── Message history ────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "📑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            chips_html = ""
            if msg.get("citations"):
                chips_html += cite_chips(msg["citations"])
            if msg.get("latency"):
                chips_html += f' <span class="dq-cite" style="color:#a78bfa; border-color:rgba(167,139,250,0.3); background:rgba(167,139,250,0.08);">⏱️ {msg["latency"]:.1f}s latency</span>'
            if chips_html:
                st.markdown(f'<div style="margin-top:8px">{chips_html}</div>', unsafe_allow_html=True)

    # ── Suggested questions on a fresh conversation ────────────────────────────
    if not st.session_state.messages:
        st.markdown('<div class="dq-suggest-label">💡 Suggested questions</div>', unsafe_allow_html=True)
        suggestions = [
            "📋 Summarize this document",
            "🔑 What are the key points?",
            "📖 What is this document about?",
            "📌 List the main topics covered",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(sug, use_container_width=True, key=f"sug_{i}"):
                    st.session_state.pending_prompt = sug
                    st.rerun()

    # ── Resolve prompt (chat box or a suggestion click) ────────────────────────
    prompt = st.chat_input(f'Ask about "{fname}"…')
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        vs = st.session_state.vector_store

        with st.chat_message("assistant", avatar="📑"):
            # ── Live "thinking" phases ─────────────────────────────────────────
            with st.status("Thinking…", expanded=True) as think:
                st.write("🔎 Searching the vector index for relevant passages…")
                try:
                    results = vs.similarity_search(query=prompt, k=st.session_state.top_k)
                except Exception as e:
                    think.update(label="Search failed", state="error")
                    st.error(f"Search failed: {friendly_error(str(e))}")
                    st.stop()

                context, citations = build_context(results)
                pages = ", ".join(f"p.{c['page']}" for c in citations) or "—"
                st.write(f"📚 Retrieved **{len(results)}** passages · pages {pages}")
                st.write("🧠 Reading the context and composing a grounded answer…")
                hist = history_msgs()
                think.update(label="Answering", state="complete", expanded=False)

            try:
                import time
                t0 = time.time()
                answer = st.write_stream(stream_answer(prompt, context, hist))
                latency = time.time() - t0
            except Exception as e:
                answer = f"⚠️ Error generating answer: {friendly_error(str(e))}"
                st.error(answer)
                latency = None
                
            chips_html = ""
            if citations:
                chips_html += cite_chips(citations)
            if latency:
                chips_html += f' <span class="dq-cite" style="color:#a78bfa; border-color:rgba(167,139,250,0.3); background:rgba(167,139,250,0.08);">⏱️ {latency:.1f}s latency</span>'
            if chips_html:
                st.markdown(f'<div style="margin-top:8px">{chips_html}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "citations": citations, "latency": latency
        })
