import os
from typing import List, Dict

from whoosh.fields import Schema, TEXT, ID, KEYWORD
from whoosh.index import create_in


SCHEMA = Schema(
    doc_id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    body=TEXT(stored=True),
    source=ID(stored=True),
    created_at=ID(stored=True),
    tags=KEYWORD(stored=True, commas=True, lowercase=True),
)


def build_whoosh_index(corpus: List[Dict], index_dir: str, rebuild: bool = True):
    if rebuild and os.path.exists(index_dir):
        # Whoosh uses files in the directory; easiest is to recreate
        for name in os.listdir(index_dir):
            path = os.path.join(index_dir, name)
            if os.path.isfile(path):
                os.remove(path)
    os.makedirs(index_dir, exist_ok=True)
    ix = create_in(index_dir, SCHEMA)

    writer = ix.writer(limitmb=256)
    for doc in corpus:
        writer.add_document(
            doc_id=str(doc.get("doc_id", "")),
            title=doc.get("title", ""),
            body=doc.get("body", ""),
            source=doc.get("source", ""),
            created_at=str(doc.get("created_at", "")),
            tags=",".join(doc.get("tags", []) or []),
        )
    writer.commit()
