#!/usr/bin/env python3
"""Build a Whoosh index from downloaded Gutenberg books."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional

from whoosh import index
from whoosh.fields import ID, NUMERIC, TEXT, Schema


def _ensure_index(index_dir: str) -> None:
    """Ensure the Whoosh index exists at the given directory."""
    if not os.path.exists(index_dir):
        os.makedirs(index_dir, exist_ok=True)
    if index.exists_in(index_dir):
        return
    schema = Schema(
        id=ID(stored=True, unique=True),
        title=TEXT(stored=True),
        authors=TEXT(stored=True),
        year=NUMERIC(stored=True),
        text=TEXT(stored=True),
    )
    index.create_in(index_dir, schema)


def _iter_metadata(metadata_path: str) -> Iterable[Dict]:
    """Yield metadata records from a JSONL file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


GUTENBERG_START_RE = re.compile(
    r"^\*{3}\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)
GUTENBERG_END_RE = re.compile(
    r"^\*{3}\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate when present."""
    lines = text.splitlines()
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    for idx, line in enumerate(lines):
        if GUTENBERG_START_RE.match(line.strip()):
            start_idx = idx + 1
            break

    for idx in range(len(lines) - 1, -1, -1):
        if GUTENBERG_END_RE.match(lines[idx].strip()):
            end_idx = idx
            break

    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx]).strip()
    return text.strip()


def _index_batch(writer, docs: List[Dict]) -> None:
    """Add a batch of documents to a Whoosh writer."""
    for doc in docs:
        writer.update_document(**doc)


def build_index(index_dir: str, data_dir: str, bulk_size: int) -> None:
    """Index downloaded Gutenberg books into Whoosh."""
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    _ensure_index(index_dir)
    ix = index.open_dir(index_dir)
    writer = ix.writer()

    batch: List[Dict] = []
    for record in _iter_metadata(metadata_path):
        text_path = record.get("text_path")
        if not text_path or not os.path.exists(text_path):
            continue
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            text = _strip_gutenberg_boilerplate(f.read())

        authors = record.get("authors") or []
        authors_text = ", ".join([a for a in authors if a])
        year = record.get("year") or 0

        doc = {
            "id": str(record.get("id")),
            "title": record.get("title", ""),
            "authors": authors_text,
            "year": year,
            "text": text,
        }
        batch.append(doc)

        if len(batch) >= bulk_size:
            _index_batch(writer, batch)
            batch = []

    if batch:
        _index_batch(writer, batch)

    writer.commit()


def main() -> None:
    """Parse CLI args and build the Whoosh index."""
    parser = argparse.ArgumentParser(description="Build a Whoosh index from Gutenberg books.")
    parser.add_argument("--index-dir", type=str, default="data/index", help="Index directory.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--bulk-size", type=int, default=100, help="Bulk request size.")
    args = parser.parse_args()

    build_index(args.index_dir, args.data_dir, args.bulk_size)
    print("Indexing complete")


if __name__ == "__main__":
    main()
