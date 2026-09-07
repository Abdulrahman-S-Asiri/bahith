"""Arabic retrieval with pinned Matryoshka embeddings and explicit ranking methods."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

MODEL_NAME = "Omartificial-Intelligence-Space/Harrier-Arabic-Matryoshka-0.6B"
MODEL_REVISION = "397a969cf4db4b9bcda254e857823d87077b7e6f"
SUPPORTED_DIMS = (64, 128, 256, 512, 768, 1024)
DEFAULT_DIM = 1024
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
DEFAULT_CORPUS = Path(__file__).parent / "corpus.json"
METHODS = ("semantic", "keyword", "hybrid", "cascade")
MAX_QUERY_CHARS = 2000


def validate_corpus(corpus: list[dict]) -> None:
    if not isinstance(corpus, list):
        raise ValueError("corpus must be a list")
    seen = set()
    for row in corpus:
        if not isinstance(row, dict) or not isinstance(row.get("id"), int) or isinstance(row["id"], bool):
            raise ValueError("each passage must have an integer id")
        if row["id"] in seen:
            raise ValueError("duplicate passage id")
        seen.add(row["id"])
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("each passage must have nonempty text")


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        corpus = json.load(stream)
    validate_corpus(corpus)
    return corpus


def lexical_tokens(text: str) -> list[str]:
    """Conservative lexical normalization only; dense inputs retain original spelling."""
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text.lower())
    text = re.sub("[أإآ]", "ا", text)
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


class ArabicSearcher:
    def __init__(self, corpus: list[dict], model_name: str = MODEL_NAME,
                 model: object | None = None, *, model_revision: str = MODEL_REVISION,
                 cache_dir: str | Path | None = None, query_prompt: str = "",
                 load_embeddings: bool = True, query_cache_size: int = 128) -> None:
        validate_corpus(corpus)
        if model is None and model_name != MODEL_NAME and model_revision == MODEL_REVISION:
            raise ValueError("a different model requires its own explicit revision")
        self.corpus = [dict(row) for row in corpus]
        self.model_name, self.model_revision, self.query_prompt = model_name, model_revision, query_prompt
        self._model, self._injected_model = model, model is not None
        self._lock = threading.RLock()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.query_cache_size = max(0, query_cache_size)
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.last_query_cached = False
        self._embeddings: np.ndarray | None = None
        self._embeddings_by_dim: dict[int, np.ndarray] = {}
        self._terms = [Counter(lexical_tokens(row["text"])) for row in self.corpus]
        self._lengths = np.array([sum(row.values()) for row in self._terms], dtype=np.float32)
        self._df = Counter(term for row in self._terms for term in row)
        if load_embeddings:
            self._ensure_embeddings()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, revision=self.model_revision,
                trust_remote_code=False, local_files_only=os.getenv("HF_HUB_OFFLINE", "0") == "1")
        return self._model

    @property
    def embeddings(self) -> np.ndarray:
        with self._lock:
            self._ensure_embeddings()
            return self._embeddings

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None or self._injected_model:
            return None
        identity = {"format": 1, "model": self.model_name, "revision": self.model_revision,
                    "document_prompt": "", "normalization": "l2-float32", "dimension": DEFAULT_DIM,
                    "text": text}
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        if not self.corpus:
            self._embeddings = np.empty((0, DEFAULT_DIM), dtype=np.float32)
            return
        vectors: dict[str, np.ndarray] = {}
        missing = []
        for row in self.corpus:
            text = row["text"]
            if text in vectors or text in missing:
                continue
            path = self._cache_path(text)
            if path and path.is_file():
                try:
                    cached = np.load(path, allow_pickle=False)
                    if cached.shape == (DEFAULT_DIM,):
                        vectors[text] = self._as_vector(cached, "cached embedding")
                except (OSError, ValueError, EOFError):
                    pass
            if text not in vectors:
                missing.append(text)
        if missing:
            encoded = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            batch = self._as_matrix(encoded, "corpus embeddings")
            if batch.shape != (len(missing), DEFAULT_DIM):
                raise ValueError(f"model produced shape {batch.shape}; expected 1024 dimensions and {len(missing)} rows")
            for text, vector in zip(missing, batch):
                vectors[text] = vector
                path = self._cache_path(text)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                    try:
                        with temporary.open("wb") as stream:
                            np.save(stream, vector, allow_pickle=False)
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
        matrix = np.vstack([vectors[row["text"]] for row in self.corpus])
        self._embeddings = matrix
        self._embeddings_by_dim[DEFAULT_DIM] = matrix

    @staticmethod
    def _validate_dim(dim: int) -> int:
        if isinstance(dim, bool) or dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {dim}")
        return dim

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("embedding contains non-finite values")
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        if np.any(norm < 1e-12) or not np.isfinite(norm).all():
            raise ValueError("embedding has a zero or invalid norm")
        return arr / norm

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
        if dim > vec.shape[-1]:
            raise ValueError(f"cannot truncate {vec.shape[-1]} dimensions to {dim}")
        return cls._normalize(vec[..., :dim])

    @staticmethod
    def _limit_top_k(top_k: int, corpus_size: int) -> int:
        return max(0, min(int(top_k), corpus_size))

    def encode_query(self, query: str) -> np.ndarray:
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query must contain 1–{MAX_QUERY_CHARS} characters")
        with self._lock:
            self.last_query_cached = query in self._query_cache
            if self.last_query_cached:
                self._query_cache.move_to_end(query)
                return self._query_cache[query].copy()
            vector = self.encode_queries([query])[0]
            if self.query_cache_size:
                self._query_cache[query] = vector.copy()
                while len(self._query_cache) > self.query_cache_size:
                    self._query_cache.popitem(last=False)
            return vector

    def clear_query_cache(self) -> None:
        with self._lock:
            self._query_cache.clear()
            self.last_query_cached = False

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, DEFAULT_DIM), dtype=np.float32)
        if any(not query.strip() or len(query.strip()) > MAX_QUERY_CHARS for query in queries):
            raise ValueError(f"query must contain 1–{MAX_QUERY_CHARS} characters")
        with self._lock:
            encoded = self.model.encode([self.query_prompt + query.strip() for query in queries],
                                        normalize_embeddings=True, show_progress_bar=False)
        vectors = self._as_matrix(encoded, "query embeddings")
        if vectors.shape != (len(queries), DEFAULT_DIM):
            raise ValueError(f"query embeddings have shape {vectors.shape}; expected {len(queries)} rows and 1024 dimensions")
        return vectors

    def _dense_scores(self, query_vector: np.ndarray, dim: int) -> np.ndarray:
        dim = self._validate_dim(dim)
        query = self._as_vector(query_vector, "query embedding")
        if query.shape != (DEFAULT_DIM,):
            raise ValueError("query embedding must have 1024 dimensions before truncation")
        self._ensure_embeddings()
        if dim not in self._embeddings_by_dim:
            self._embeddings_by_dim[dim] = self._truncate(self._embeddings, dim)
        return np.clip(self._embeddings_by_dim[dim] @ self._truncate(query, dim), -1.0, 1.0)

    def _indices(self, document_id: str | None = None) -> np.ndarray:
        return np.array([i for i, row in enumerate(self.corpus)
                         if not document_id or row.get("document_id") == document_id], dtype=int)

    def _hits(self, scores: np.ndarray, indices: np.ndarray, top_k: int, kind: str) -> list[dict]:
        limit = self._limit_top_k(top_k, len(indices))
        ordered = indices[np.lexsort((indices, -scores[indices]))[:limit]]
        return [{**self.corpus[i], "score": float(scores[i]), "score_kind": kind} for i in ordered]

    def search_by_vector(self, query_vector: np.ndarray, top_k: int = DEFAULT_TOP_K,
                         dim: int = DEFAULT_DIM, document_id: str | None = None) -> list[dict]:
        self._validate_dim(dim)
        indices = self._indices(document_id)
        if not len(indices) or top_k <= 0:
            return []
        with self._lock:
            return self._hits(self._dense_scores(query_vector, dim), indices, top_k, "cosine")

    def keyword_search(self, query: str, top_k: int = DEFAULT_TOP_K,
                       document_id: str | None = None) -> list[dict]:
        indices = self._indices(document_id)
        if not len(indices) or not query.strip() or top_k <= 0:
            return []
        # Standard BM25, k1=1.5, b=.75, collection-wide document frequencies.
        scores = np.zeros(len(self.corpus), dtype=np.float32)
        average = max(float(self._lengths.mean()), 1.0)
        for term in set(lexical_tokens(query)):
            df = self._df[term]
            if not df:
                continue
            idf = np.log(1 + (len(self.corpus) - df + .5) / (df + .5))
            tf = np.array([self._terms[i][term] for i in indices], dtype=np.float32)
            scores[indices] += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self._lengths[indices] / average))
        return self._hits(scores, indices[scores[indices] > 0], top_k, "bm25")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, dim: int = DEFAULT_DIM,
                 method: str = "semantic", candidate_k: int = 50,
                 document_id: str | None = None) -> list[dict]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self._validate_dim(dim)
        if not query.strip() or top_k <= 0 or not len(self._indices(document_id)):
            return []
        if len(query.strip()) > MAX_QUERY_CHARS:
            raise ValueError("query is too long")
        if method == "keyword":
            self.last_query_cached = False
            return self.keyword_search(query, top_k, document_id)
        with self._lock:
            vector = self.encode_query(query)
            if method == "semantic":
                return self.search_by_vector(vector, top_k, dim, document_id)
            depth = max(top_k, int(candidate_k))
            if method == "cascade":
                candidates = self.search_by_vector(vector, depth, dim, document_id)
                ids = {row["id"] for row in candidates}
                indices = np.array([i for i, row in enumerate(self.corpus) if row["id"] in ids], dtype=int)
                scores = np.zeros(len(self.corpus), dtype=np.float32)
                scores[indices] = np.clip(self.embeddings[indices] @ self._as_vector(vector, "query embedding"), -1, 1)
                return self._hits(scores, indices, top_k, "cosine")
            dense = self.search_by_vector(vector, depth, dim, document_id)
            lexical = self.keyword_search(query, depth, document_id)
            fused: dict[int, float] = {}
            for ranked in (dense, lexical):
                for rank, row in enumerate(ranked, 1):
                    fused[row["id"]] = fused.get(row["id"], 0.0) + 1.0 / (60 + rank)
            indices = np.array([i for i, row in enumerate(self.corpus) if row["id"] in fused], dtype=int)
            scores = np.array([fused.get(row["id"], 0.0) for row in self.corpus])
            return self._hits(scores, indices, top_k, "rrf")

    def explain_vector(self, query_vector: np.ndarray, dim: int = DEFAULT_DIM, buckets: int = 16) -> dict:
        if buckets < 1:
            raise ValueError("buckets must be positive")
        vector = self._truncate(self._as_vector(query_vector, "query embedding"), dim)
        parts = np.array_split(vector, min(buckets, dim))
        values = [float(np.linalg.norm(part)) for part in parts]
        peak = max(values) or 1.0
        ranges, start = [], 1
        for part, value in zip(parts, values):
            end = start + len(part) - 1
            ranges.append({"start": start, "end": end, "value": value, "height": value / peak * 100})
            start = end + 1
        return {"dim": dim, "norm": float(np.linalg.norm(vector)), "buckets": ranges}

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, dim: int = DEFAULT_DIM) -> list[dict]:
        return self.retrieve(query, top_k, dim)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arabic Matryoshka retrieval")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dim", type=int, choices=SUPPORTED_DIMS, default=DEFAULT_DIM)
    parser.add_argument("--method", choices=METHODS, default="semantic")
    args = parser.parse_args()
    searcher = ArabicSearcher(load_corpus(args.corpus), load_embeddings=False)
    for hit in searcher.retrieve(" ".join(args.query), dim=args.dim, method=args.method):
        print(f"[{hit['score']:.4f} {hit['score_kind']}] {hit['text']}")
