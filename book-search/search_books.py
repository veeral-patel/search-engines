#!/usr/bin/env python3
"""Search the local Whoosh index for Gutenberg books."""

from __future__ import annotations

import argparse
import os

from whoosh import index

from search_utils import build_parser, build_searcher


def search_index(index_dir: str, query_text: str, limit: int) -> None:
    """Search the Whoosh index and print top results."""
    if not index.exists_in(index_dir):
        raise FileNotFoundError(f"Index not found in: {index_dir}")

    ix = index.open_dir(index_dir)
    parser = build_parser(ix)
    query = parser.parse(query_text)

    with build_searcher(ix) as searcher:
        results = searcher.search(query, limit=limit)
        results.fragmenter.charlimit = 300
        results.formatter.maxchars = 300
        for hit in results:
            title = hit.get("title", "")
            score = hit.score
            print(f"- {title} (score={score:.4f})")


def main() -> None:
    """Parse CLI args and run a search query."""
    parser = argparse.ArgumentParser(description="Search the Whoosh book index.")
    parser.add_argument("--index-dir", type=str, default="data/index", help="Index directory.")
    parser.add_argument("--query", type=str, required=True, help="Query text.")
    parser.add_argument("--limit", type=int, default=5, help="Max results to show.")
    args = parser.parse_args()

    search_index(args.index_dir, args.query, args.limit)


if __name__ == "__main__":
    main()
