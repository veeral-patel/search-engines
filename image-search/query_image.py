import argparse
from typing import List, Tuple

import duckdb
import numpy as np

from clip_utils import encode_image, load_clip


def load_db_embeddings(con: duckdb.DuckDBPyConnection) -> Tuple[List[int], List[str], np.ndarray]:
    rows = con.execute("SELECT id, path, embedding FROM images ORDER BY id").fetchall()
    if not rows:
        raise SystemExit("No embeddings in DB. Run embed_images.py first.")
    ids, paths, embeds = zip(*rows)
    matrix = np.array(embeds, dtype=np.float32)
    return list(ids), list(paths), matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Query similar images by image.")
    parser.add_argument("image", type=str, help="Path to query image")
    parser.add_argument("--db", type=str, default="data/embeddings.duckdb")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)
    ids, paths, matrix = load_db_embeddings(con)
    con.close()

    bundle = load_clip(model_name=args.model, pretrained=args.pretrained)
    query_vec = encode_image(bundle, args.image)

    scores = matrix @ query_vec
    top_k = min(args.top_k, len(scores))
    top_idx = np.argsort(-scores)[:top_k]

    print("Top matches:")
    for rank, i in enumerate(top_idx, start=1):
        print(f"{rank:02d}. score={scores[i]:.4f} id={ids[i]} path={paths[i]}")


if __name__ == "__main__":
    main()
