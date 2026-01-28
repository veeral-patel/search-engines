#!/usr/bin/env python3
"""Streamlit UI for searching the Whoosh book index."""

from __future__ import annotations

import os
import json
import subprocess

import streamlit as st
from whoosh import index
from whoosh.highlight import HtmlFormatter
from search_utils import build_parser, build_searcher


def _open_index(index_dir: str):
    if not index.exists_in(index_dir):
        return None
    return index.open_dir(index_dir)

def _load_metadata(metadata_path: str):
    if not os.path.exists(metadata_path):
        return []
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> None:
    st.set_page_config(page_title="Book Search", page_icon="📚", layout="wide")
    st.title("Book Search")
    st.caption("Search Project Gutenberg books indexed with Whoosh.")

    with st.sidebar:
        st.header("Settings")
        index_dir = st.text_input("Index directory", value="data/index")
        limit = st.number_input("Max results", min_value=1, max_value=50, value=10, step=1)
        page = st.radio("Page", ["Search", "Library"], index=0)
        if st.button("Rebuild index"):
            with st.spinner("Rebuilding index..."):
                cmd = ["python3", "build_index.py", "--index-dir", index_dir, "--data-dir", "data"]
                result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Index rebuilt successfully.")
            else:
                st.error("Index rebuild failed.")
                if result.stderr:
                    st.code(result.stderr)

    if page == "Library":
        metadata_path = os.path.join("data", "metadata.jsonl")
        records = _load_metadata(metadata_path)
        st.subheader("Downloaded books")
        st.caption(f"{len(records)} books found in {metadata_path}")
        if not records:
            st.info("No metadata found. Download books first.")
            return
        table = [
            {
                "id": r.get("id"),
                "title": r.get("title", ""),
                "authors": ", ".join(r.get("authors") or []),
                "year": r.get("year") or "",
            }
            for r in records
        ]
        st.dataframe(table, use_container_width=True)
        return

    query_text = st.text_input("Search query", value="")

    if not query_text.strip():
        st.info("Enter a query to search the index.")
        return

    ix = _open_index(index_dir)
    if ix is None:
        st.error(f"Index not found in: {index_dir}")
        return

    parser = build_parser(ix)
    query = parser.parse(query_text)

    with build_searcher(ix) as searcher:
        results = searcher.search(query, limit=int(limit))
        results.formatter = HtmlFormatter(tagname="mark", classname="match")
        st.write(f"Found {len(results)} results")
        for hit in results:
            title = hit.get("title", "")
            authors = hit.get("authors", "")
            year = hit.get("year", 0)
            score = hit.score
            title_snippet = hit.highlights("title") or title
            authors_snippet = hit.highlights("authors") or authors
            snippet = hit.highlights("text") or hit.get("text", "")[:300]

            st.markdown(f"**{title_snippet}** ({year})", unsafe_allow_html=True)
            st.markdown(
                f"<span class='authors'>{authors_snippet} · score {score:.4f}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(snippet, unsafe_allow_html=True)
            st.divider()


if __name__ == "__main__":
    main()
