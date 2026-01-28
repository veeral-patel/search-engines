import json
import os
from typing import Dict, Iterable, List


DEFAULT_CONFIG = {
    "embedder": "hashing",
    "embedding_dim": 384,
    "whoosh_index_dir": "whoosh_index",
    "duckdb_path": "vectors.duckdb",
    "corpus_path": "data/corpus.jsonl",
}


def load_config(path: str) -> Dict:
    if not os.path.exists(path):
        return dict(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(data)
    return cfg


def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
