"""Streamlit Interactive Web UI for Intelligent Knowledge Assistant."""

import streamlit as st
import httpx
import time
from typing import Dict, Any, List

# Page Configuration
st.set_page_config(
    page_title="Intelligent Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-rag {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-agent {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-refusal {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_health() -> Dict[str, Any]:
    try:
        res = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        return res.json() if res.status_code == 200 else {}
    except Exception:
        return {}


def query_assistant(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        res = httpx.post(f"{API_BASE_URL}/query", json=payload, timeout=60.0)
        if res.status_code == 200:
            return res.json()
        return {"error": f"API Error ({res.status_code}): {res.text}"}
    except Exception as e:
        return {"error": f"Failed to connect to API server at {API_BASE_URL}: {str(e)}"}


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric-vectors/512/artificial-intelligence.png", width=70)
    st.title("System Status")
    health = fetch_health()
    if health.get("status") == "healthy":
        st.success(f"🟢 API Online (v{health.get('version', '0.1.0')})")
        st.caption(f"📚 Indexed Chunks: **{health.get('total_chunks', 0)}**")
        st.markdown("**Indexed Corpus:**")
        for doc in health.get("documents", []):
            st.caption(f"• `{doc}`")
    else:
        st.warning("🟡 API Service Offline or Re-indexing")
        st.caption("Start the backend with `poetry run uvicorn src.api.app:app --port 8000`")

    st.divider()
    st.subheader("⚙️ Query Configuration")
    workflow_mode = st.selectbox(
        "Routing Mode",
        options=["auto", "simple_rag", "agentic_workflow"],
        format_func=lambda x: {
            "auto": "🤖 Dynamic Auto-Router",
            "simple_rag": "⚡ Force Simple RAG",
            "agentic_workflow": "🔄 Force Agentic Workflow",
        }[x],
    )
    doc_filter_opt = st.selectbox(
        "Document Filter (Optional)",
        options=["None", "rag_architecture_patterns.pdf", "vector_database_comparison.pdf", "agentic_ai_frameworks.pdf"],
    )
    doc_filter = None if doc_filter_opt == "None" else doc_filter_opt
    top_k = st.slider("Retrieval Chunks (Top-K)", min_value=2, max_value=10, value=5)


# Header
st.markdown('<div class="main-title">🧠 Intelligent Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Dual-mode RAG & Multi-Step Agentic Assistant strictly grounded in technical architecture reference documents with verifiable citations.</div>',
    unsafe_allow_html=True,
)

# Preset Demo Queries
st.markdown("##### 💡 Try Sample Evaluation Queries:")
col1, col2, col3, col4 = st.columns(4)

preset_query = None
with col1:
    if st.button("📌 1. Simple Factoid RAG", use_container_width=True):
        preset_query = "What chunk size and overlap are recommended for dense QA in RAG?"
with col2:
    if st.button("⚖️ 2. Multi-Doc Comparison", use_container_width=True):
        preset_query = "Compare Pinecone and Chroma on latency, cost, and persistence."
with col3:
    if st.button("🧩 3. Agentic Multi-Hop", use_container_width=True):
        preset_query = "How do LangGraph and AutoGen compare in their orchestration primitives?"
with col4:
    if st.button("🚫 4. Out-of-Corpus Refusal", use_container_width=True):
        preset_query = "What is the best recipe for baking chocolate brownies?"


# Chat / Input Form
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg and msg["meta"]:
            meta = msg["meta"]
            wf = meta.get("workflow", "")
            badge = "badge-rag" if "SIMPLE" in wf else ("badge-agent" if "AGENT" in wf else "badge-refusal")
            st.markdown(f'<span class="{badge}">{wf}</span> &nbsp; Latency: <b>{meta.get("latency_seconds", 0)}s</b>', unsafe_allow_html=True)

user_input = st.chat_input("Ask a technical question about RAG, Vector DBs, or Agent Frameworks...")

# Handle trigger from preset buttons or chat input
active_query = preset_query or user_input

if active_query:
    # Render user query
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Execute assistant call
    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and retrieving grounded context..."):
            req_payload = {
                "query": active_query,
                "workflow_override": workflow_mode,
                "doc_filter": doc_filter,
                "top_k": top_k,
            }
            res = query_assistant(req_payload)

        if "error" in res:
            st.error(res["error"])
        else:
            ans = res.get("answer", "")
            wf = res.get("workflow", "UNKNOWN")
            latency = res.get("latency_seconds", 0.0)
            citations = res.get("citations", [])
            steps = res.get("steps")
            reasoning = res.get("routing_reasoning", "")

            # Badge styling
            badge_class = "badge-rag" if wf == "SIMPLE_RAG" else ("badge-agent" if wf == "AGENTIC_WORKFLOW" else "badge-refusal")
            st.markdown(
                f'<span class="{badge_class}">Workflow: {wf}</span> &nbsp; Latency: <b>{latency}s</b>',
                unsafe_allow_html=True,
            )

            if reasoning:
                st.caption(f"🧠 **Routing Logic**: {reasoning}")

            # Answer
            st.markdown(ans)

            # Agent Steps Expander
            if steps:
                with st.expander(f"🔍 Agent Execution Trace ({len(steps)} Steps)", expanded=False):
                    for s in steps:
                        st.markdown(f"**Step {s['step_number']}**: `{s['tool_name']}` ({s['latency_seconds']}s)")
                        st.markdown(f"*Thought*: {s['thought']}")
                        if s.get("tool_input"):
                            st.json(s["tool_input"])
                        if s.get("observation"):
                            st.text_area(f"Observation (Step {s['step_number']})", s["observation"][:400], height=80, disabled=True)
                        st.divider()

            # Citations Expander
            if citations:
                with st.expander(f"📑 Sources & Citations ({len(citations)})", expanded=True):
                    for c in citations:
                        st.markdown(f"- 📄 **{c['document_name']}** — Page **{c['page_number']}** (`{c['raw_tag']}`)")

            # Save in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": ans,
                "meta": {"workflow": wf, "latency_seconds": latency},
            })
