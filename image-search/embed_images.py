import argparse
from pathlib import Path

import duckdb
import numpy as np
from tqdm import tqdm

from clip_utils import encode_image, load_clip


def ensure_schema(con: duckdb.DuckDBPyConnection, reset: bool) -> None:
    if reset:
        con.execute("DROP TABLE IF EXISTS images")
        con.execute("DROP TABLE IF EXISTS metadata")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER,
            path TEXT,
            embedding FLOAT[]
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def set_metadata(con: duckdb.DuckDBPyConnection, key: str, value: str) -> None:
    con.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", [key, value]
    )


def get_metadata(con: duckdb.DuckDBPyConnection, key: str) -> str | None:
    row = con.execute("SELECT value FROM metadata WHERE key = ?", [key]).fetchone()
    return row[0] if row else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed images with OpenCLIP.")
    parser.add_argument("--image-dir", type=str, default="data/images")
    parser.add_argument("--db", type=str, default="data/embeddings.duckdb")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not paths:
        raise SystemExit(f"No images found in {image_dir}")

    con = duckdb.connect(args.db)
    ensure_schema(con, args.reset)

    existing_model = get_metadata(con, "model_name")
    existing_pretrained = get_metadata(con, "pretrained")
    if existing_model and existing_model != args.model:
        raise SystemExit(
            f"DB model mismatch: {existing_model} vs {args.model}. Use --reset to rebuild."
        )
    if existing_pretrained and existing_pretrained != args.pretrained:
        raise SystemExit(
            f"DB pretrained mismatch: {existing_pretrained} vs {args.pretrained}. Use --reset to rebuild."
        )

    set_metadata(con, "model_name", args.model)
    set_metadata(con, "pretrained", args.pretrained)

    bundle = load_clip(model_name=args.model, pretrained=args.pretrained)

    con.execute("DELETE FROM images")
    for idx, path in enumerate(tqdm(paths, desc="Embedding")):
        embedding = encode_image(bundle, str(path))
        con.execute(
            "INSERT INTO images (id, path, embedding) VALUES (?, ?, ?)",
            [idx, str(path), embedding.tolist()],
        )

    con.close()


if __name__ == "__main__":
    main()
