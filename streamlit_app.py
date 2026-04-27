"""
Codebase Q&A Assistant — Fully Self-Contained Streamlit App
============================================================
No FastAPI backend required. All services (cloning, chunking,
embedding, vector store, RAG) run directly inside Streamlit.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Set your GROQ_API_KEY in a .env file or in .streamlit/secrets.toml:
    GROQ_API_KEY = "gsk_..."
"""

import os
import shutil
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Codebase Q&A Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }

input[type="text"] {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
.answer-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #7c3aed;
    border-radius: 8px;
    padding: 18px 22px;
    color: #e6edf3;
    font-size: 15px;
    line-height: 1.8;
    margin-top: 10px;
    white-space: pre-wrap;
}
.chip-row { margin-top: 10px; }
.chip {
    display: inline-block;
    background: #21262d;
    color: #a78bfa;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-family: monospace;
    margin: 3px 3px 3px 0;
}
.badge {
    background: #7c3aed;
    color: white;
    border-radius: 12px;
    padding: 3px 12px;
    font-size: 13px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 8px;
}
.metric-row { display: flex; gap: 12px; margin-top: 12px; }
.metric-tile {
    flex: 1;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.metric-num   { font-size: 26px; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 12px; color: #8b949e; margin-top: 2px; }
.q-bubble {
    background: #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    color: #e6edf3;
    margin-bottom: 6px;
    font-weight: 500;
}
.divider-line { border: none; border-top: 1px solid #21262d; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SERVICE LAYER  (replaces app/services/*)
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
REPOS_DIR       = os.getenv("REPOS_DIR", "./data/repos")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/vectorstores")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 4

VALID_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}
IGNORED_DIRS     = {
    ".git", "node_modules", "venv", "dist",
    "__pycache__", "build", ".idea", ".vscode",
    "env", ".env", "site-packages",
}

CODE_SEPARATORS = ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]

PROMPT_TEMPLATE = """You are a senior software engineer doing a code review.
Your job is to answer questions about a codebase using ONLY the code context provided below.

Rules:
- Answer based strictly on the context. Do not guess or hallucinate.
- If the answer is not present in the context, say: "I don't know based on the provided code."
- Be concise and technical. Reference specific functions, classes, or files when relevant.
- Format code snippets with triple backticks when showing code.

Context (relevant code chunks):
{context}

Question: {question}

Answer:"""


# ── Embeddings (cached for the entire Streamlit session) ─────────────────────
@st.cache_resource(show_spinner="Loading embedding model (first time ~30 s)…")
def get_embeddings():
    """Cache the HuggingFace embedding model for the whole session."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── GitHub loader ─────────────────────────────────────────────────────────────
def clone_repository(repo_url: str):
    """Clone a public GitHub repo and return (Documents, repo_name)."""
    from git import Repo, GitCommandError
    from langchain_core.documents import Document

    if not repo_url.startswith("https://github.com/"):
        raise ValueError("Invalid GitHub URL. Must start with https://github.com/")

    repo_name  = repo_url.rstrip("/").split("/")[-1]
    clone_path = Path(REPOS_DIR) / repo_name

    if clone_path.exists():
        shutil.rmtree(clone_path)

    try:
        logger.info(f"Cloning {repo_url} → {clone_path}")
        Repo.clone_from(repo_url, str(clone_path), depth=1)
    except GitCommandError as e:
        raise ValueError(
            f"Failed to clone repository. Make sure it's public and the URL is correct.\n{e}"
        )

    documents = _load_code_files(clone_path, repo_name)
    if not documents:
        raise ValueError(
            f"Repository has no supported code files ({', '.join(VALID_EXTENSIONS)})."
        )

    logger.info(f"Loaded {len(documents)} files from '{repo_name}'")
    return documents, repo_name


def _load_code_files(base_path: Path, repo_name: str):
    from langchain_core.documents import Document

    documents = []
    for file_path in base_path.rglob("*"):
        if file_path.is_dir():
            continue
        if any(part in IGNORED_DIRS for part in file_path.parts):
            continue
        if file_path.suffix not in VALID_EXTENSIONS:
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue
            relative_path = str(file_path.relative_to(base_path))
            documents.append(Document(
                page_content=content,
                metadata={
                    "filename":  file_path.name,
                    "filepath":  relative_path,
                    "repo":      repo_name,
                    "extension": file_path.suffix,
                },
            ))
        except Exception as e:
            logger.warning(f"Skipping {file_path}: {e}")

    return documents


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CODE_SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Chunking: {len(documents)} files → {len(chunks)} chunks")
    return chunks


# ── Vector store ──────────────────────────────────────────────────────────────
def _store_path(repo_name: str) -> str:
    return str(Path(VECTORSTORE_DIR) / repo_name)


def build_vectorstore(chunks, embeddings, repo_name):
    from langchain_community.vectorstores import FAISS

    path = _store_path(repo_name)
    os.makedirs(path, exist_ok=True)
    logger.info(f"Building FAISS index for '{repo_name}' ({len(chunks)} chunks)…")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(path)
    logger.info(f"FAISS index saved → {path}")
    return vs


def load_vectorstore(embeddings, repo_name):
    from langchain_community.vectorstores import FAISS

    path = _store_path(repo_name)
    if not Path(path).exists():
        return None
    return FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )


# ── RAG pipeline ──────────────────────────────────────────────────────────────
def answer_question(query: str, vectorstore, groq_api_key: str) -> dict:
    """Full RAG pipeline: retrieve → prompt → Groq Llama 3."""
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0,
        groq_api_key=groq_api_key,
        max_tokens=1024,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa_chain.invoke({"query": query})
    sources = list({
        doc.metadata.get("filepath", "unknown")
        for doc in result.get("source_documents", [])
    })
    return {"answer": result["result"], "sources": sources}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "repo_name"    not in st.session_state: st.session_state.repo_name    = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "repo_stats"   not in st.session_state: st.session_state.repo_stats   = {}
if "vectorstore"  not in st.session_state: st.session_state.vectorstore  = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Codebase Q&A")
    st.markdown("Ask natural language questions about any public GitHub repository.")
    st.divider()

    # ── API Key input ──────────────────────────────────────────────────────
    groq_key_env = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        value=groq_key_env,
        type="password",
        placeholder="gsk_…  (free at console.groq.com)",
        help="Your key is never stored — it lives only in this browser session.",
    )

    st.divider()

    if st.session_state.repo_name:
        st.success(f"✅ Active: `{st.session_state.repo_name}`")
        stats = st.session_state.repo_stats
        if stats:
            st.markdown(
                f"""
                <div class="metric-row">
                  <div class="metric-tile">
                    <div class="metric-num">{stats.get('files_processed', '—')}</div>
                    <div class="metric-label">Files</div>
                  </div>
                  <div class="metric-tile">
                    <div class="metric-num">{stats.get('chunks_created', '—')}</div>
                    <div class="metric-label">Chunks</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("🔄 Load a different repo", use_container_width=True):
            st.session_state.repo_name    = None
            st.session_state.chat_history = []
            st.session_state.repo_stats   = {}
            st.session_state.vectorstore  = None
            st.rerun()
    else:
        st.info("No repository loaded yet. Use Step 1 to get started.")

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("""
- 🦙 Groq Llama 3 70B (LLM)
- 🤗 MiniLM-L6-v2 (Embeddings)
- 🗄️ FAISS (Vector DB)
- 🐙 GitPython (Cloning)
- 🦜 LangChain (RAG)
    """)
    st.divider()
    st.caption("Built with ❤️ using Streamlit + LangChain")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 Codebase Q&A Assistant")
st.markdown("Understand any GitHub codebase by asking questions in plain English.")
st.markdown('<hr class="divider-line">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Repository
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<span class="badge">Step 1</span> &nbsp; Load a GitHub Repository', unsafe_allow_html=True)
st.markdown("")

repo_url = st.text_input(
    label="GitHub Repository URL",
    placeholder="https://github.com/username/repository-name",
    disabled=bool(st.session_state.repo_name),
)

col_btn, col_hint = st.columns([1, 5])
with col_btn:
    process_btn = st.button(
        "🚀 Process",
        use_container_width=True,
        disabled=bool(st.session_state.repo_name),
    )
with col_hint:
    st.caption("Only public GitHub repos are supported. Large repos may take 2–4 minutes.")

if process_btn:
    if not groq_api_key.strip():
        st.error("⚠️ Enter your Groq API key in the sidebar first.")
    elif not repo_url.strip():
        st.error("⚠️ Please enter a GitHub URL.")
    elif not repo_url.startswith("https://github.com/"):
        st.error("⚠️ URL must start with https://github.com/")
    else:
        os.makedirs(REPOS_DIR, exist_ok=True)
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)

        progress = st.progress(0, text="Initialising…")
        status   = st.empty()

        try:
            status.info("📥 Cloning repository from GitHub…")
            progress.progress(10, text="Cloning…")
            documents, repo_name = clone_repository(repo_url)

            progress.progress(35, text="Chunking code files…")
            status.info("✂️ Chunking source files…")
            chunks = chunk_documents(documents)

            progress.progress(55, text="Loading embedding model…")
            status.info("🤗 Loading embedding model (cached after first use)…")
            embeddings = get_embeddings()

            progress.progress(70, text="Building FAISS index…")
            status.info("🧮 Generating embeddings and building FAISS index…")
            vectorstore = build_vectorstore(chunks, embeddings, repo_name)

            progress.progress(100, text="Done!")
            status.empty()
            progress.empty()

            st.session_state.repo_name   = repo_name
            st.session_state.vectorstore = vectorstore
            st.session_state.repo_stats  = {
                "files_processed": len(documents),
                "chunks_created":  len(chunks),
            }
            st.session_state.chat_history = []

            st.success(
                f"✅ **{repo_name}** is ready! "
                f"{len(documents)} files · {len(chunks)} chunks indexed."
            )
            st.rerun()

        except ValueError as e:
            progress.empty(); status.empty()
            st.error(f"❌ {e}")
        except Exception as e:
            progress.empty(); status.empty()
            st.error(f"Unexpected error: {e}")

st.markdown('<hr class="divider-line">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Ask Questions
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<span class="badge">Step 2</span> &nbsp; Ask Questions About the Code', unsafe_allow_html=True)
st.markdown("")

if not st.session_state.repo_name:
    st.info("💡 Complete Step 1 first — load a repository to enable the Q&A.")
else:
    # ── Lazy-load vectorstore if missing from session (e.g. after page refresh) ──
    if st.session_state.vectorstore is None:
        embeddings = get_embeddings()
        st.session_state.vectorstore = load_vectorstore(
            embeddings, st.session_state.repo_name
        )

    # ── Example chips ────────────────────────────────────────────────────────
    st.caption("💡 Example questions:")
    ex_cols = st.columns(4)
    examples = [
        "What does this project do?",
        "How is authentication handled?",
        "What are the main API endpoints?",
        "How is the database connected?",
    ]
    selected_example = None
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                selected_example = ex

    # ── Question input ────────────────────────────────────────────────────────
    query = st.text_input(
        label="Your question",
        placeholder="e.g. How does error handling work in this codebase?",
        value=selected_example if selected_example else "",
        key="query_input",
    )

    ask_col, clear_col = st.columns([1, 1])
    with ask_col:
        ask_btn = st.button("💬 Ask", use_container_width=True, type="primary")
    with clear_col:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if ask_btn:
        if not groq_api_key.strip():
            st.warning("⚠️ Enter your Groq API key in the sidebar.")
        elif not query.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("🔍 Searching code and generating answer…"):
                try:
                    result = answer_question(
                        query,
                        st.session_state.vectorstore,
                        groq_api_key,
                    )
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer":   result["answer"],
                        "sources":  result.get("sources", []),
                    })
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Chat history ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
        st.markdown("### 💬 Conversation")

        for item in reversed(st.session_state.chat_history):
            st.markdown(
                f'<div class="q-bubble">🧑 {item["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="answer-card">🤖 {item["answer"]}</div>',
                unsafe_allow_html=True,
            )
            if item["sources"]:
                chips = "".join(
                    f'<span class="chip">📄 {s}</span>' for s in item["sources"]
                )
                st.markdown(
                    f'<div class="chip-row">'
                    f'<strong style="color:#8b949e;font-size:12px;">SOURCES &nbsp;</strong>'
                    f'{chips}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
