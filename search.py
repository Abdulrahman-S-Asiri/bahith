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
DEFAULT_CORPUS = Path(__file__).parent / "corpus.json"


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class ArabicSearcher:
    def __init__(self, corpus: list[dict], model_name: str = MODEL_NAME) -> None:
        self.corpus = corpus
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        # Encode the whole corpus once at full 1024-dim. Per-query truncation
        # to a smaller Matryoshka dim happens inside `search`.
        self.embeddings = self.model.encode(
            [doc["text"] for doc in corpus],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @staticmethod
    def _truncate(vec: np.ndarray, dim: int) -> np.ndarray:
        if dim >= vec.shape[-1]:
            return vec
        # Cosine similarity assumes unit-length vectors. Slicing breaks that
        # invariant, so we re-normalise after the cut.
        sliced = vec[..., :dim]
        norm = np.linalg.norm(sliced, axis=-1, keepdims=True)
        return sliced / np.clip(norm, 1e-12, None)

    def search(self, query: str, top_k: int = 5, dim: int = 1024) -> list[dict]:
        if dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {dim}")
        q_vec = self.model.encode(
            query, normalize_embeddings=True, show_progress_bar=False,
        )
        q_vec = self._truncate(q_vec, dim)
        doc_vecs = self._truncate(self.embeddings, dim)
        scores = doc_vecs @ q_vec
        top_idx = np.argsort(-scores)[:top_k]
        return [{**self.corpus[i], "score": float(scores[i])} for i in top_idx]


def _cli() -> None:
    query = " ".join(sys.argv[1:]).strip() or "ما فوائد ممارسة الرياضة؟"
    print(f"Loading model: {MODEL_NAME}")
    t0 = time.perf_counter()
    searcher = ArabicSearcher(load_corpus())
    print(f"Ready in {time.perf_counter() - t0:.1f}s · {len(searcher.corpus)} docs\n")
    print(f"Query: {query}\n")
    for i, hit in enumerate(searcher.search(query, top_k=5), 1):
        print(f"  {i}. [{hit['score']:.3f}] ({hit['category']}) {hit['text']}")


if __name__ == "__main__":
    _cli()
