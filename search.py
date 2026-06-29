"""Arabic semantic search backed by Harrier-Arabic-Matryoshka-0.6B."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Omartificial-Intelligence-Space/Harrier-Arabic-Matryoshka-0.6B"
SUPPORTED_DIMS = (64, 128, 256, 512, 768, 1024)
DEFAULT_DIM = max(SUPPORTED_DIMS)
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
DEFAULT_CORPUS = Path(__file__).parent / "corpus.json"


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class ArabicSearcher:
    def __init__(
        self,
        corpus: list[dict],
        model_name: str = MODEL_NAME,
        model: object | None = None,
    ) -> None:
        self.corpus = corpus
        self.model = model or SentenceTransformer(model_name, trust_remote_code=True)
        encoded = self.model.encode(
            [doc["text"] for doc in corpus],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = self._as_matrix(encoded, "corpus embeddings")
        if self.embeddings.shape[-1] != DEFAULT_DIM:
            raise ValueError(
                f"model produced {self.embeddings.shape[-1]} dimensions; "
                f"expected {DEFAULT_DIM}"
            )
        self._embeddings_by_dim = {
            dim: self._truncate(self.embeddings, dim) for dim in SUPPORTED_DIMS
        }

    @staticmethod
    def _validate_dim(dim: int) -> int:
        if dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {dim}")
        return dim

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec / np.clip(norm, 1e-12, None)

    @classmethod
    def _as_vector(cls, vec: np.ndarray, label: str) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"{label} must be one vector, got shape {arr.shape}")
        return cls._normalize(arr)

    @classmethod
    def _as_matrix(cls, vec: np.ndarray, label: str) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{label} must be a 2D matrix, got shape {arr.shape}")
        return cls._normalize(arr)

    @classmethod
    def _truncate(cls, vec: np.ndarray, dim: int) -> np.ndarray:
        cls._validate_dim(dim)
        return cls._normalize(vec[..., :dim])

    @staticmethod
    def _limit_top_k(top_k: int, corpus_size: int) -> int:
        return max(0, min(int(top_k), corpus_size))

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, DEFAULT_DIM), dtype=np.float32)
        encoded = self.model.encode(
            [query.strip() for query in queries],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_vecs = self._as_matrix(encoded, "query embeddings")
        if q_vecs.shape[-1] != DEFAULT_DIM:
            raise ValueError(
                f"query embeddings have {q_vecs.shape[-1]} dimensions; expected {DEFAULT_DIM}"
            )
        return q_vecs

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        dim: int = DEFAULT_DIM,
    ) -> list[dict]:
        dim = self._validate_dim(dim)
        limit = self._limit_top_k(top_k, len(self.corpus))
        if limit == 0:
            return []

        q_vec = self._truncate(self._as_vector(query_vector, "query embedding"), dim)
        scores = np.clip(self._embeddings_by_dim[dim] @ q_vec, -1.0, 1.0)
        top_idx = np.lexsort((np.arange(len(scores)), -scores))[:limit]
        return [{**self.corpus[i], "score": float(scores[i])} for i in top_idx]

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        dim: int = DEFAULT_DIM,
    ) -> list[dict]:
        if not query.strip():
            return []
        return self.search_by_vector(self.encode_query(query), top_k=top_k, dim=dim)


def _cli() -> None:
    query = " ".join(sys.argv[1:]).strip() or "ما فوائد ممارسة الرياضة؟"
    print(f"Loading model: {MODEL_NAME}")
    t0 = time.perf_counter()
    searcher = ArabicSearcher(load_corpus())
    print(f"Ready in {time.perf_counter() - t0:.1f}s · {len(searcher.corpus)} docs\n")
    print(f"Query: {query}\n")
    for i, hit in enumerate(searcher.search(query, top_k=DEFAULT_TOP_K), 1):
        print(f"  {i}. [{hit['score']:.3f}] ({hit['category']}) {hit['text']}")


if __name__ == "__main__":
    _cli()
