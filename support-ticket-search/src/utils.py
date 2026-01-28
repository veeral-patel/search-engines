import hashlib
import math
import re
from typing import Iterable, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def stable_hash(token: str) -> int:
    # Deterministic hash across runs and machines
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def hashing_embed(text: str, dim: int) -> List[float]:
    vec = [0.0] * dim
    for tok in tokenize(text):
        idx = stable_hash(tok) % dim
        vec[idx] += 1.0
    return l2_normalize(vec)


def minmax_norm(results: Iterable[tuple]) -> dict:
    results = list(results)
    if not results:
        return {}
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {doc_id: 1.0 for doc_id, _ in results}
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in results}


def snippet(text: str, length: int = 160) -> str:
    if text is None:
        return ""
    text = " ".join(text.split())
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."
