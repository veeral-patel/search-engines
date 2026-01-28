from typing import Dict, List, Tuple

import duckdb
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser

from .utils import minmax_norm


def bm25_search(query: str, index_dir: str, k: int) -> List[Tuple[str, float]]:
    ix = open_dir(index_dir)
    with ix.searcher() as searcher:
        parser = MultifieldParser(["title", "body"], schema=ix.schema)
        q = parser.parse(query)
        results = searcher.search(q, limit=k)
        return [(r["doc_id"], r.score) for r in results]


def _try_load_vss(con) -> bool:
    try:
        con.execute("LOAD vss;")
        return True
    except Exception:
        try:
            con.execute("INSTALL vss;")
            con.execute("LOAD vss;")
            return True
        except Exception:
            return False


def vector_search(query_embedding: List[float], duckdb_path: str, k: int) -> List[Tuple[str, float]]:
    con = duckdb.connect(duckdb_path)
    use_vss = _try_load_vss(con)

    if use_vss:
        rows = con.execute(
            """
            SELECT doc_id, list_distance(embedding, ?) AS dist
            FROM doc_vectors
            ORDER BY dist ASC
            LIMIT ?
            """,
            [query_embedding, k],
        ).fetchall()
        return [(doc_id, 1.0 / (1.0 + dist)) for doc_id, dist in rows]

    raise RuntimeError("DuckDB vss extension is required for vector search.")


def blend_scores(
    bm25_results: List[Tuple[str, float]],
    vec_results: List[Tuple[str, float]],
    w_bm25: float,
    w_vec: float,
) -> List[Tuple[str, float, float, float]]:
    bm25_norm = minmax_norm(bm25_results)
    vec_norm = minmax_norm(vec_results)

    all_ids = set(bm25_norm) | set(vec_norm)
    blended = []
    for doc_id in all_ids:
        b = bm25_norm.get(doc_id, 0.0)
        v = vec_norm.get(doc_id, 0.0)
        score = w_bm25 * b + w_vec * v
        blended.append((doc_id, score, b, v))
    blended.sort(key=lambda x: x[1], reverse=True)
    return blended


def rrf_blend(
    bm25_results: List[Tuple[str, float]],
    vec_results: List[Tuple[str, float]],
    k: int = 60,
) -> List[Tuple[str, float, float, float]]:
    scores: Dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(vec_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    blended = [(doc_id, score, 0.0, 0.0) for doc_id, score in scores.items()]
    blended.sort(key=lambda x: x[1], reverse=True)
    return blended
