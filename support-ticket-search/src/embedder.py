from typing import List, Optional

from .utils import hashing_embed


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        return hashing_embed(text, self.dim)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: Optional[int] = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()


def load_embedder(embedder: str, dim: int, model_name: Optional[str] = None):
    if embedder == "hashing":
        return HashingEmbedder(dim=dim)
    if embedder == "sentence-transformers":
        if not model_name:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model_name=model_name, dim=dim)
    raise ValueError(f"Unknown embedder: {embedder}")
