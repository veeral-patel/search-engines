#!/usr/bin/env python3
"""Download Project Gutenberg books as plain text with metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET

GUTENDEX_URL = "https://gutendex.com/books"
GUTENBERG_RDF_URL = "https://www.gutenberg.org/ebooks/{book_id}.rdf"

TEXT_FORMAT_PREFERENCES = [
    "text/plain; charset=utf-8",
    "text/plain; charset=us-ascii",
    "text/plain",
]


def _safe_filename(name: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    if len(cleaned) > max_len:
        return cleaned[:max_len].rstrip("_")
    return cleaned


def _pick_text_url(formats: Dict[str, str]) -> Optional[str]:
    for key in TEXT_FORMAT_PREFERENCES:
        if key in formats:
            return formats[key]
    # fall back to any text/plain key
    for key, value in formats.items():
        if key.startswith("text/plain"):
            return value
    return None


def _fetch_gutendex_page(page: int) -> Dict:
    resp = requests.get(GUTENDEX_URL, params={"page": page}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _extract_rdf_year(book_id: int) -> Optional[int]:
    try:
        resp = requests.get(GUTENBERG_RDF_URL.format(book_id=book_id), timeout=30)
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
        ns = {
            "dcterms": "http://purl.org/dc/terms/",
        }
        issued = root.find(".//dcterms:issued", ns)
        if issued is None or issued.text is None:
            return None
        # issued is usually YYYY-MM-DD
        year_match = re.match(r"(\d{4})", issued.text.strip())
        if not year_match:
            return None
        return int(year_match.group(1))
    except Exception:
        return None


def _download_text(url: str, dest_path: str) -> None:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def _iter_books(target_count: int) -> Iterable[Dict]:
    page = 1
    collected = 0
    while collected < target_count:
        data = _fetch_gutendex_page(page)
        for book in data.get("results", []):
            yield book
            collected += 1
            if collected >= target_count:
                return
        page += 1


def download_books(count: int, out_dir: str, sleep_s: float) -> Tuple[int, int]:
    os.makedirs(out_dir, exist_ok=True)
    books_dir = os.path.join(out_dir, "books")
    os.makedirs(books_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.jsonl")

    downloaded = 0
    skipped = 0

    with open(metadata_path, "w", encoding="utf-8") as meta_out:
        for book in _iter_books(count):
            book_id = book.get("id")
            title = book.get("title") or ""
            authors = [a.get("name") for a in book.get("authors", []) if a.get("name")]
            text_url = _pick_text_url(book.get("formats", {}))

            if not book_id or not text_url:
                skipped += 1
                continue

            filename = f"{book_id}_{_safe_filename(title) or 'book'}.txt"
            text_path = os.path.join(books_dir, filename)

            if os.path.exists(text_path):
                skipped += 1
                continue

            _download_text(text_url, text_path)
            year = _extract_rdf_year(book_id)

            record = {
                "id": book_id,
                "title": title,
                "authors": authors,
                "year": year,
                "text_path": text_path,
            }
            meta_out.write(json.dumps(record, ensure_ascii=True) + "\n")

            downloaded += 1
            if sleep_s:
                time.sleep(sleep_s)

    return downloaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Project Gutenberg books as plain text.")
    parser.add_argument("--count", type=int, default=100, help="Number of books to download.")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory for books and metadata.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between downloads (seconds).")
    args = parser.parse_args()

    downloaded, skipped = download_books(args.count, args.out_dir, args.sleep)
    print(f"Downloaded: {downloaded}, skipped: {skipped}")


if __name__ == "__main__":
    main()
