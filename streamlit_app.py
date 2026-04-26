"""
Codebase Q&A Assistant — Standalone Streamlit App
===================================================
No FastAPI backend needed. All logic runs directly in Streamlit.

Run: streamlit run streamlit_app.py

Requirements (install once):
    pip install streamlit gitpython langchain langchain-community \
                langchain-huggingface langchain-groq sentence-transformers \
                faiss-cpu python-dotenv groq
"""

import os
import shutil
import logging
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Codebase Q&A Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs",
                    ".jsx", ".tsx", ".rb", ".php", ".swift", ".kt", ".scala"}
IGNORED_DIRS = {
    ".git", "node_modules", "venv", "dist", "__pycache__",
    "build", ".idea", ".vscode", "env", ".env", "site-packages",
}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CODE_SEPARATORS = ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]

PROMPT_TEMPLATE = """\
You are a senior software engineer doing a code review.
Answer questions about a codebase using ONLY the code context below.

Rules:
- Answer strictly from the context. Do not guess or hallucinate.
- If the answer is not in the context, say: "I don't know based on the provided code."
- Be concise and technical. Reference specific functions, classes, or files when relevant.
- Format code snippets with triple backticks.

Context (relevant code chunks):
{context}

Question: {question}

Answer:"""

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
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
.metric-num  { font-size: 26px; font-weight: 700; color: #a78bfa; }
.metric-label{ font-size: 12px; color: #8b949e; margin-top: 2px; }
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
# INLINE SERVICE FUNCTIONS  (replaces FastAPI backend)
# ══════════════════════════════════════════════════════════════════════════════

def _load_code_files(base_path: Path, repo_name: str):
    """Recursively read all valid code files from a cloned repo."""
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
                    "filename": file_path.name,
                    "filepath": relative_path,
                    "repo": repo_name,
                    "extension": file_path.suffix,
                },
            ))
        except Exception as e:
            logger.warning(f"Skipping {file_path}: {e}")
    return documents


def clone_and_load(repo_url: str):
    """Clone a public GitHub repo into a temp dir and load code files."""
    from git import Repo, GitCommandError

    if not repo_url.startswith("https://github.com/"):
        raise ValueError("URL must start with https://github.com/")

    repo_name = repo_url.rstrip("/").split("/")[-1]
    tmp_dir = Path(tempfile.mkdtemp()) / repo_name

    try:
        Repo.clone_from(repo_url, str(tmp_dir), depth=1)
    except GitCommandError as e:
        raise ValueError(f"Clone failed — is the repo public? Details: {e}")

    documents = _load_code_files(tmp_dir, repo_name)
    if not documents:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(
            f"No supported code files found ({', '.join(sorted(VALID_EXTENSIONS))})."
        )
    return documents, repo_name, tmp_dir


def chunk_documents(documents):
    """Split documents into overlapping chunks for embedding."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CODE_SEPARATORS,
        length_function=len,
    )
    return splitter.split_documents(documents)


@st.cache_resource(show_spinner="Loading embedding model (one-time ~30s)…")
def get_embeddings():
    """Load and cache the HuggingFace embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks, embeddings):
    """Build an in-memory FAISS vectorstore from chunks."""
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(chunks, embeddings)


def answer_question(query: str, vectorstore, groq_api_key: str) -> dict:
    """Run the full RAG pipeline and return answer + source files."""
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

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

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Retrieve docs first so we can return sources
    retrieved_docs = retriever.invoke(query)
    context = "

".join(doc.page_content for doc in retrieved_docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    sources = list({
        doc.metadata.get("filepath", "unknown")
        for doc in retrieved_docs
    })

    return {"answer": answer, "sources": sources}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for key, default in [
    ("repo_name", None),
    ("chat_history", []),
    ("repo_stats", {}),
    ("vectorstore", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Codebase Q&A")
    st.markdown("Ask natural language questions about any public GitHub repository.")
    st.divider()

    # ── API Key input ──────────────────────────────────────────────────────
    st.markdown("**🔑 Groq API Key**")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        help="Free key at https://console.groq.com",
    )
    if not groq_key:
        # Try env/secrets as fallback
        groq_key = (
            os.environ.get("GROQ_API_KEY")
            or st.secrets.get("GROQ_API_KEY", "")
        )
    if groq_key:
        st.success("API key provided ✅")
    else:
        st.warning("Enter your Groq API key to enable Q&A.")
    st.divider()

    # ── Repo status ────────────────────────────────────────────────────────
    if st.session_state.repo_name:
        st.success(f"✅ Active: `{st.session_state.repo_name}`")
        stats = st.session_state.repo_stats
        if stats:
            st.markdown(
                f"""
                <div class="metric-row">
                  <div class="metric-tile">
                    <div class="metric-num">{stats.get('files', '—')}</div>
                    <div class="metric-label">Files</div>
                  </div>
                  <div class="metric-tile">
                    <div class="metric-num">{stats.get('chunks', '—')}</div>
                    <div class="metric-label">Chunks</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("🔄 Load a different repo", use_container_width=True):
            st.session_state.repo_name = None
            st.session_state.chat_history = []
            st.session_state.repo_stats = {}
            st.session_state.vectorstore = None
            st.rerun()
    else:
        st.info("No repository loaded yet. Use Step 1 to get started.")

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("""
- 🦙 Groq Llama 3 70B (LLM)
- 🤗 MiniLM-L6-v2 (Embeddings)
- 🗄️ FAISS in-memory (Vector DB)
- 🐙 GitPython (Cloning)
- ⚡ 100% Streamlit — no backend
    """)
    st.divider()
    st.caption("Built with Streamlit · No localhost required")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 Codebase Q&A Assistant")
st.markdown("Understand any GitHub codebase by asking questions in plain English.")
st.markdown('<hr class="divider-line">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Repository
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<span class="badge">Step 1</span> &nbsp; Load a GitHub Repository',
            unsafe_allow_html=True)
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
    if not repo_url.strip():
        st.error("⚠️ Please enter a GitHub URL.")
    elif not repo_url.startswith("https://github.com/"):
        st.error("⚠️ URL must start with https://github.com/")
    elif not groq_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
    else:
        progress_bar = st.progress(0, text="Initialising…")
        status = st.empty()

        try:
            status.info("📥 Cloning repository from GitHub…")
            progress_bar.progress(10, text="Cloning repo…")

            documents, repo_name, tmp_dir = clone_and_load(repo_url)

            status.info("✂️ Chunking code files…")
            progress_bar.progress(40, text="Chunking code…")

            chunks = chunk_documents(documents)

            status.info("🧮 Loading embedding model…")
            progress_bar.progress(55, text="Loading embedder…")

            embeddings = get_embeddings()

            status.info("📐 Building FAISS vector index…")
            progress_bar.progress(75, text="Building index…")

            vectorstore = build_vectorstore(chunks, embeddings)

            progress_bar.progress(100, text="Done!")
            status.empty()
            progress_bar.empty()

            st.session_state.repo_name = repo_name
            st.session_state.vectorstore = vectorstore
            st.session_state.repo_stats = {
                "files": len(documents),
                "chunks": len(chunks),
            }
            st.session_state.chat_history = []

            # Clean up cloned files (vectorstore is in-memory)
            shutil.rmtree(tmp_dir, ignore_errors=True)

            st.success(
                f"✅ **{repo_name}** is ready! "
                f"{len(documents)} files · {len(chunks)} chunks indexed."
            )
            st.rerun()

        except ValueError as e:
            progress_bar.empty()
            status.empty()
            st.error(f"❌ {e}")
        except Exception as e:
            progress_bar.empty()
            status.empty()
            st.error(f"❌ Unexpected error: {e}")

st.markdown('<hr class="divider-line">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Ask Questions
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<span class="badge">Step 2</span> &nbsp; Ask Questions About the Code',
            unsafe_allow_html=True)
st.markdown("")

if not st.session_state.repo_name:
    st.info("💡 Complete Step 1 first — load a repository to enable Q&A.")
else:
    # Example chips
    st.caption("💡 Example questions:")
    examples = [
        "What does this project do?",
        "How is authentication handled?",
        "What are the main API endpoints?",
        "How is the database connected?",
    ]
    ex_cols = st.columns(4)
    selected_example = None
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                selected_example = ex

    query = st.text_input(
        label="Your question",
        placeholder="e.g. How does error handling work in this codebase?",
        value=selected_example or "",
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
        if not query.strip():
            st.warning("Please type a question first.")
        elif not groq_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar.")
        elif st.session_state.vectorstore is None:
            st.error("⚠️ Vector store not found — please reload the repository.")
        else:
            with st.spinner("🔍 Searching code and generating answer…"):
                try:
                    result = answer_question(
                        query,
                        st.session_state.vectorstore,
                        groq_key,
                    )
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Chat history ───────────────────────────────────────────────────────
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
