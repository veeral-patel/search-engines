import os
import sys
from typing import Dict

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from src.embedder import load_embedder  # noqa: E402
from src.search import bm25_search, blend_scores, rrf_blend, vector_search  # noqa: E402
from src.storage import load_config, read_jsonl  # noqa: E402
from src.utils import snippet  # noqa: E402
from src.cli import rerank  # noqa: E402


@st.cache_data
def load_docs(corpus_path: str) -> Dict[str, Dict]:
    docs = read_jsonl(corpus_path)
    return {str(d.get("doc_id", "")): d for d in docs}


@st.cache_resource
def get_embedder(name: str, dim: int, model: str):
    return load_embedder(name, dim=dim, model_name=model)


st.set_page_config(page_title="Support Ticket Search", layout="wide")

st.title("Support Ticket Search")

cfg = load_config(os.path.join(ROOT, "config.json"))
corpus_path = os.path.join(ROOT, cfg["corpus_path"])
index_dir = os.path.join(ROOT, cfg["whoosh_index_dir"])
duckdb_path = os.path.join(ROOT, cfg["duckdb_path"])

with st.sidebar:
    st.header("Search Settings")
    embedder_name = st.selectbox("Embedder", ["hashing", "sentence-transformers"], index=0)
    model_name = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    top_n = st.number_input("Top N", min_value=1, max_value=50, value=10)
    blend = st.selectbox("Blend", ["weighted", "rrf"], index=0)
    w_bm25 = st.slider("BM25 weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    w_vec = st.slider("Vector weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    use_rerank = st.checkbox("Rerank (cross-encoder)", value=False)
    reranker_model = st.text_input("Reranker model", value="cross-encoder/ms-marco-MiniLM-L-6-v2")

page = st.sidebar.radio("Page", ["Search", "All Tickets"], index=0)

if page == "Search":
    query = st.text_input("Query", value="rotate s3 access key")

    if st.button("Search", type="primary"):
        if not os.path.exists(corpus_path):
            st.error("Corpus not found. Run ingest first.")
            st.stop()

        docs = load_docs(corpus_path)
        embedder = get_embedder(embedder_name, cfg.get("embedding_dim", 384), model_name)

        with st.spinner("Running hybrid search..."):
            try:
                bm25 = bm25_search(query, index_dir=index_dir, k=50)
                qvec = embedder.embed(query)
                vec = vector_search(qvec, duckdb_path=duckdb_path, k=50)
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                st.stop()

        if blend == "rrf":
            blended = rrf_blend(bm25, vec, k=60)
        else:
            blended = blend_scores(bm25, vec, w_bm25=float(w_bm25), w_vec=float(w_vec))

            if use_rerank:
                blended = rerank(blended, query, corpus_path, reranker_model)

        st.subheader("Results")
        for rank, (doc_id, score, b, v) in enumerate(blended[: int(top_n)], start=1):
            doc = docs.get(doc_id, {})
            title = doc.get("title", "")
            body = doc.get("body", "")

            st.markdown(f"**{rank:02d}. {doc_id}** — score: `{score:.4f}`")
            if not use_rerank:
                st.caption(f"bm25: {b:.4f} | vec: {v:.4f}")
            st.write(title)
            st.write(snippet(body))
            with st.expander("Details"):
                st.json(doc)

if page == "All Tickets":
    if not os.path.exists(corpus_path):
        st.error("Corpus not found. Run ingest first.")
        st.stop()

    docs = load_docs(corpus_path)
    tickets = [d for d in docs.values() if d.get("source") == "ticket"]

    st.subheader(f"All Tickets ({len(tickets)})")
    keyword = st.text_input("Filter by keyword", value="")
    if keyword:
        kw = keyword.lower()
        tickets = [
            d
            for d in tickets
            if kw in (d.get("title", "").lower() + " " + d.get("body", "").lower())
        ]

    for doc in tickets:
        st.markdown(f"**{doc.get('doc_id','')}** — {doc.get('title','')}")
        st.write(snippet(doc.get("body", "")))
        with st.expander("Details"):
            st.json(doc)
