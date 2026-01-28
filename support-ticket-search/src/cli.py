import argparse
import os
import sys
from typing import Dict, List

import duckdb

from .embedder import load_embedder
from .indexer import build_whoosh_index
from .search import bm25_search, blend_scores, rrf_blend, vector_search
from .storage import load_config, read_jsonl, write_jsonl
from .utils import snippet


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def cmd_ingest(args, cfg, base_dir: str):
    corpus_path = resolve_path(base_dir, cfg["corpus_path"])
    merged: Dict[str, Dict] = {}
    for input_path in args.input:
        items = read_jsonl(input_path)
        for doc in items:
            doc_id = str(doc.get("doc_id", "")).strip()
            if not doc_id:
                continue
            merged[doc_id] = doc
    write_jsonl(corpus_path, merged.values())
    print(f"Wrote {len(merged)} docs to {corpus_path}")


def cmd_index(args, cfg, base_dir: str):
    corpus_path = resolve_path(base_dir, cfg["corpus_path"])
    index_dir = resolve_path(base_dir, cfg["whoosh_index_dir"])
    corpus = read_jsonl(corpus_path)
    if not corpus:
        print(f"No docs found in {corpus_path}")
        return
    build_whoosh_index(corpus, index_dir, rebuild=args.rebuild)
    print(f"Indexed {len(corpus)} docs into {index_dir}")


def _create_vectors_table(con, dim: int, rebuild: bool):
    if rebuild:
        con.execute("DROP TABLE IF EXISTS doc_vectors")
    con.execute(f"CREATE TABLE IF NOT EXISTS doc_vectors (doc_id TEXT PRIMARY KEY, embedding FLOAT[{dim}])")


def cmd_embed(args, cfg, base_dir: str):
    corpus_path = resolve_path(base_dir, cfg["corpus_path"])
    duckdb_path = resolve_path(base_dir, cfg["duckdb_path"])
    corpus = read_jsonl(corpus_path)
    if not corpus:
        print(f"No docs found in {corpus_path}")
        return

    embedder_name = args.embedder or cfg.get("embedder", "hashing")
    dim = args.dim or cfg.get("embedding_dim", 384)
    embedder = load_embedder(embedder_name, dim=dim, model_name=args.model)

    con = duckdb.connect(duckdb_path)

    # Create table based on actual embedding size
    sample_vec = embedder.embed((corpus[0].get("title", "") + "\n" + corpus[0].get("body", "")).strip())
    dim = len(sample_vec)
    _create_vectors_table(con, dim, rebuild=args.rebuild)

    rows: List[tuple] = []
    rows.append((str(corpus[0].get("doc_id", "")), sample_vec))
    for doc in corpus[1:]:
        text = (doc.get("title", "") + "\n" + doc.get("body", "")).strip()
        rows.append((str(doc.get("doc_id", "")), embedder.embed(text)))

    con.executemany("INSERT OR REPLACE INTO doc_vectors VALUES (?, ?)", rows)

    if args.vss_index:
        try:
            con.execute("INSTALL vss;")
            con.execute("LOAD vss;")
            con.execute("CREATE INDEX IF NOT EXISTS doc_vectors_idx ON doc_vectors USING vss(embedding)")
            print("Built vss index")
        except Exception as exc:
            print(f"Could not build vss index: {exc}")

    print(f"Embedded {len(rows)} docs into {duckdb_path}")


def _load_doc_texts(corpus_path: str) -> Dict[str, Dict]:
    docs = read_jsonl(corpus_path)
    return {str(d.get("doc_id", "")): d for d in docs}


def cmd_search(args, cfg, base_dir: str):
    index_dir = resolve_path(base_dir, cfg["whoosh_index_dir"])
    duckdb_path = resolve_path(base_dir, cfg["duckdb_path"])
    corpus_path = resolve_path(base_dir, cfg["corpus_path"])

    embedder_name = args.embedder or cfg.get("embedder", "hashing")
    dim = args.dim or cfg.get("embedding_dim", 384)
    embedder = load_embedder(embedder_name, dim=dim, model_name=args.model)

    bm25 = bm25_search(args.query, index_dir=index_dir, k=args.k)
    qvec = embedder.embed(args.query)
    vec = vector_search(qvec, duckdb_path=duckdb_path, k=args.k)

    if args.blend == "rrf":
        blended = rrf_blend(bm25, vec, k=args.rrf_k)
    else:
        blended = blend_scores(bm25, vec, w_bm25=args.w_bm25, w_vec=args.w_vec)

    if args.rerank:
        blended = rerank(blended, args.query, corpus_path, args.reranker_model)

    docs = _load_doc_texts(corpus_path)
    top = blended[: args.top_n]

    print("\nResults:\n")
    for rank, (doc_id, score, b, v) in enumerate(top, start=1):
        doc = docs.get(doc_id, {})
        title = doc.get("title", "")
        body = doc.get("body", "")
        print(f"{rank:02d}. {doc_id} | score={score:.4f} bm25={b:.4f} vec={v:.4f}")
        print(f"    {title}")
        print(f"    {snippet(body)}\n")


def rerank(blended, query: str, corpus_path: str, model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        print("Rerank requested but sentence-transformers is not installed.")
        print("Install with: pip install sentence-transformers")
        return blended

    docs = _load_doc_texts(corpus_path)
    texts = []
    ids = []
    for doc_id, _, _, _ in blended:
        doc = docs.get(doc_id, {})
        text = (doc.get("title", "") + "\n" + doc.get("body", "")).strip()
        texts.append((query, text))
        ids.append(doc_id)

    model = CrossEncoder(model_name)
    scores = model.predict(texts)
    rescored = list(zip(ids, scores))
    rescored.sort(key=lambda x: x[1], reverse=True)

    # Return in blended tuple format (bm25/vec placeholders)
    return [(doc_id, float(score), 0.0, 0.0) for doc_id, score in rescored]


def cmd_eval(args, cfg, base_dir: str):
    corpus_path = resolve_path(base_dir, cfg["corpus_path"])
    queries_path = resolve_path(base_dir, args.queries)

    queries = read_jsonl(queries_path)
    if not queries:
        print(f"No queries found in {queries_path}")
        return

    index_dir = resolve_path(base_dir, cfg["whoosh_index_dir"])
    duckdb_path = resolve_path(base_dir, cfg["duckdb_path"])

    embedder_name = args.embedder or cfg.get("embedder", "hashing")
    dim = args.dim or cfg.get("embedding_dim", 384)
    embedder = load_embedder(embedder_name, dim=dim, model_name=args.model)

    total_rr = 0.0
    hit = 0

    for item in queries:
        query = item.get("query", "")
        relevant = set(item.get("relevant", []))

        bm25 = bm25_search(query, index_dir=index_dir, k=args.k)
        qvec = embedder.embed(query)
        vec = vector_search(qvec, duckdb_path=duckdb_path, k=args.k)

        if args.blend == "rrf":
            blended = rrf_blend(bm25, vec, k=args.rrf_k)
        else:
            blended = blend_scores(bm25, vec, w_bm25=args.w_bm25, w_vec=args.w_vec)

        top_ids = [doc_id for doc_id, _, _, _ in blended[: args.top_n]]

        rr = 0.0
        for idx, doc_id in enumerate(top_ids, start=1):
            if doc_id in relevant:
                rr = 1.0 / idx
                break
        if rr > 0:
            hit += 1
        total_rr += rr

    mrr = total_rr / len(queries)
    recall_at_k = hit / len(queries)

    print(f"MRR@{args.top_n}: {mrr:.4f}")
    print(f"Recall@{args.top_n}: {recall_at_k:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Support Ticket Search CLI")
    parser.add_argument("--config", default="config.json")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Merge JSONL files into corpus")
    p_ingest.add_argument("--input", action="append", required=True)
    p_ingest.set_defaults(func=cmd_ingest)

    p_index = sub.add_parser("index", help="Build Whoosh BM25 index")
    p_index.add_argument("--rebuild", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_embed = sub.add_parser("embed", help="Create embeddings in DuckDB")
    p_embed.add_argument("--rebuild", action="store_true")
    p_embed.add_argument("--embedder", choices=["hashing", "sentence-transformers"])
    p_embed.add_argument("--model", help="SentenceTransformer model name")
    p_embed.add_argument("--dim", type=int)
    p_embed.add_argument("--vss-index", action="store_true")
    p_embed.set_defaults(func=cmd_embed)

    p_search = sub.add_parser("search", help="Hybrid search")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=50, help="candidate pool size")
    p_search.add_argument("--top-n", type=int, default=10)
    p_search.add_argument("--blend", choices=["weighted", "rrf"], default="weighted")
    p_search.add_argument("--w-bm25", type=float, default=0.6)
    p_search.add_argument("--w-vec", type=float, default=0.4)
    p_search.add_argument("--rrf-k", type=int, default=60)
    p_search.add_argument("--embedder", choices=["hashing", "sentence-transformers"])
    p_search.add_argument("--model", help="SentenceTransformer model name")
    p_search.add_argument("--dim", type=int)
    p_search.add_argument("--rerank", action="store_true")
    p_search.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="Evaluate on labeled queries")
    p_eval.add_argument("--queries", default="data/queries.jsonl")
    p_eval.add_argument("--k", type=int, default=50)
    p_eval.add_argument("--top-n", type=int, default=10)
    p_eval.add_argument("--blend", choices=["weighted", "rrf"], default="weighted")
    p_eval.add_argument("--w-bm25", type=float, default=0.6)
    p_eval.add_argument("--w-vec", type=float, default=0.4)
    p_eval.add_argument("--rrf-k", type=int, default=60)
    p_eval.add_argument("--embedder", choices=["hashing", "sentence-transformers"])
    p_eval.add_argument("--model", help="SentenceTransformer model name")
    p_eval.add_argument("--dim", type=int)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))

    args.func(args, cfg, base_dir)


if __name__ == "__main__":
    main()
