from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from search import DEFAULT_DIM, SUPPORTED_DIMS, ArabicSearcher, load_corpus, validate_corpus


def full_vector(*values: float, dim: int = DEFAULT_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[: len(values)] = values
    return vec


def prefix_cosine(left: np.ndarray, right: np.ndarray, dim: int) -> float:
    """Scalar reference calculation, independent of the searcher's vector helpers."""
    products = math.fsum(float(left[i]) * float(right[i]) for i in range(dim))
    left_norm = math.sqrt(math.fsum(float(left[i]) ** 2 for i in range(dim)))
    right_norm = math.sqrt(math.fsum(float(right[i]) ** 2 for i in range(dim)))
    return products / (left_norm * right_norm)


class FakeModel:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors
        self.encoded_inputs: list[object] = []

    def encode(
        self,
        values: str | list[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.encoded_inputs.append(values)
        if isinstance(values, list):
            return np.vstack([self.vectors[value] for value in values])
        return self.vectors[values]


class ArabicSearcherMathTests(unittest.TestCase):
    def make_searcher(self) -> tuple[ArabicSearcher, FakeModel]:
        corpus = [
            {"id": 1, "category": "test", "text": "doc-a"},
            {"id": 2, "category": "test", "text": "doc-b"},
            {"id": 3, "category": "test", "text": "doc-c"},
        ]
        model = FakeModel(
            {
                "doc-a": full_vector(1.0, 0.0),
                "doc-b": full_vector(1.0, 0.0),
                "doc-c": full_vector(0.0, 1.0),
                "query": full_vector(1.0, 0.0),
            }
        )
        return ArabicSearcher(corpus, model=model), model

    def test_truncate_renormalizes_rows(self) -> None:
        matrix = np.zeros((2, DEFAULT_DIM), dtype=np.float32)
        matrix[0, :2] = [3.0, 4.0]
        matrix[1, :2] = [1.0, 1.0]

        truncated = ArabicSearcher._truncate(matrix, 64)

        self.assertEqual(truncated.shape, (2, 64))
        np.testing.assert_allclose(np.linalg.norm(truncated, axis=1), [1.0, 1.0])
        np.testing.assert_allclose(truncated[0, :2], [0.6, 0.8], atol=1e-6)

    def test_all_supported_dimensions_match_independent_matryoshka_oracle(self) -> None:
        query = np.array([1.0 + (i % 7) / 10 for i in range(DEFAULT_DIM)], dtype=np.float32)
        doc_a = np.array([0.5 + (i % 11) / 7 for i in range(DEFAULT_DIM)], dtype=np.float32)
        doc_b = np.array([1.8 - (i % 13) / 12 for i in range(DEFAULT_DIM)], dtype=np.float32)
        corpus = [
            {"id": 1, "category": "test", "text": "doc-a"},
            {"id": 2, "category": "test", "text": "doc-b"},
        ]
        searcher = ArabicSearcher(corpus, model=FakeModel({"doc-a": doc_a, "doc-b": doc_b}))

        for dim in SUPPORTED_DIMS:
            with self.subTest(dim=dim):
                hits = searcher.search_by_vector(query, top_k=2, dim=dim)
                actual = {hit["id"]: hit["score"] for hit in hits}
                expected = {
                    1: prefix_cosine(query, doc_a, dim),
                    2: prefix_cosine(query, doc_b, dim),
                }
                self.assertAlmostEqual(actual[1], expected[1], places=6)
                self.assertAlmostEqual(actual[2], expected[2], places=6)
                self.assertEqual([hit["id"] for hit in hits], sorted(expected, key=lambda i: (-expected[i], i)))
                if dim < DEFAULT_DIM:
                    self.assertGreater(float(np.linalg.norm(query[dim:])), 0.0)
                    unrenormalized = float(
                        np.dot(query[:dim], doc_a[:dim])
                        / (np.linalg.norm(query) * np.linalg.norm(doc_a))
                    )
                    self.assertGreater(abs(actual[1] - unrenormalized), 1e-3)

    def test_truncate_rejects_invalid_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "dim must be one of"):
            ArabicSearcher._truncate(np.ones((1, DEFAULT_DIM), dtype=np.float32), 63)

        tail_only = full_vector()
        tail_only[64] = 1.0
        with self.assertRaisesRegex(ValueError, "zero or invalid norm"):
            ArabicSearcher._truncate(tail_only, 64)

    def test_nonfinite_and_zero_embeddings_are_rejected(self) -> None:
        corpus = [{"id": 1, "category": "test", "text": "doc"}]
        invalid = {
            "zero": full_vector(),
            "nan": full_vector(float("nan")),
            "infinity": full_vector(float("inf")),
        }
        for label, vector in invalid.items():
            with self.subTest(corpus_embedding=label):
                with self.assertRaises(ValueError):
                    ArabicSearcher(corpus, model=FakeModel({"doc": vector}))

        for label, vector in invalid.items():
            with self.subTest(query_embedding=label):
                model = FakeModel({"doc": full_vector(1.0), "query": vector})
                searcher = ArabicSearcher(corpus, model=model)
                with self.assertRaises(ValueError):
                    searcher.search("query", dim=64)

    def test_model_dimension_is_validated(self) -> None:
        corpus = [{"id": 1, "category": "test", "text": "doc"}]
        model = FakeModel({"doc": full_vector(1.0, dim=768)})

        with self.assertRaisesRegex(ValueError, r"\(1, 768\).*1024 dimensions"):
            ArabicSearcher(corpus, model=model)

    def test_search_orders_by_score_then_corpus_order(self) -> None:
        searcher, _ = self.make_searcher()

        hits = searcher.search("query", top_k=3, dim=64)

        self.assertEqual([hit["id"] for hit in hits], [1, 2, 3])
        np.testing.assert_allclose([hit["score"] for hit in hits], [1.0, 1.0, 0.0])

    def test_query_cache_survives_dimension_changes(self) -> None:
        searcher, model = self.make_searcher()
        initial_calls = len(model.encoded_inputs)

        searcher.retrieve("query", dim=64)
        self.assertFalse(searcher.last_query_cached)
        searcher.retrieve("query", dim=768)

        self.assertTrue(searcher.last_query_cached)
        self.assertEqual(model.encoded_inputs[initial_calls:], [["query"]])

    def test_search_limits_top_k_to_available_results(self) -> None:
        searcher, _ = self.make_searcher()

        self.assertEqual(len(searcher.search("query", top_k=99, dim=64)), 3)
        self.assertEqual(searcher.search("query", top_k=0, dim=64), [])

    def test_blank_query_does_not_encode_query(self) -> None:
        searcher, model = self.make_searcher()
        calls_before = len(model.encoded_inputs)

        self.assertEqual(searcher.search("   ", top_k=3, dim=64), [])
        self.assertEqual(len(model.encoded_inputs), calls_before)

    def test_encode_queries_batches_inputs(self) -> None:
        searcher, model = self.make_searcher()

        vectors = searcher.encode_queries(["query", "query"])

        self.assertEqual(vectors.shape, (2, DEFAULT_DIM))
        self.assertEqual(model.encoded_inputs[-1], ["query", "query"])

    def test_explain_vector_groups_dimension_magnitudes(self) -> None:
        searcher, _ = self.make_searcher()
        vector = np.zeros(DEFAULT_DIM, dtype=np.float32)
        vector[0] = 3.0
        vector[20] = 4.0

        explanation = searcher.explain_vector(vector, dim=64, buckets=4)

        self.assertEqual(explanation["dim"], 64)
        self.assertAlmostEqual(explanation["norm"], 1.0)
        self.assertEqual(
            [(b["start"], b["end"]) for b in explanation["buckets"]],
            [(1, 16), (17, 32), (33, 48), (49, 64)],
        )
        np.testing.assert_allclose(
            [b["value"] for b in explanation["buckets"]],
            [0.6, 0.8, 0.0, 0.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            [b["height"] for b in explanation["buckets"]],
            [75.0, 100.0, 0.0, 0.0],
            atol=1e-6,
        )

    def test_search_rejects_bad_query_dimension(self) -> None:
        searcher, _ = self.make_searcher()

        with self.assertRaises(ValueError):
            searcher.search_by_vector(full_vector(1.0, dim=64), top_k=1, dim=1024)

    def test_keyword_search_does_not_load_dense_model(self) -> None:
        corpus = [
            {"id": 1, "text": "العِلْم نور"},
            {"id": 2, "text": "نص آخر"},
        ]
        searcher = ArabicSearcher(corpus, load_embeddings=False)

        hits = searcher.retrieve("العلم", method="keyword", top_k=2)

        self.assertEqual([hit["id"] for hit in hits], [1])
        self.assertEqual(hits[0]["score_kind"], "bm25")
        self.assertIsNone(searcher._model)
        self.assertIsNone(searcher._embeddings)

    def test_hybrid_uses_reciprocal_rank_fusion(self) -> None:
        corpus = [
            {"id": 1, "text": "تفاح"},
            {"id": 2, "text": "تفاح تفاح"},
            {"id": 3, "text": "موز"},
        ]
        model = FakeModel(
            {
                "تفاح": full_vector(1.0, 0.0),
                "تفاح تفاح": full_vector(0.0, 1.0),
                "موز": full_vector(0.8, 0.6),
            }
        )
        searcher = ArabicSearcher(corpus, model=model)

        hits = searcher.retrieve("تفاح", method="hybrid", top_k=3, dim=64)

        self.assertEqual([hit["id"] for hit in hits], [1, 2, 3])
        expected = {
            1: 1 / 61 + 1 / 62,
            2: 1 / 63 + 1 / 61,
            3: 1 / 62,
        }
        for hit in hits:
            self.assertEqual(hit["score_kind"], "rrf")
            self.assertAlmostEqual(hit["score"], expected[hit["id"]])

    def test_cascade_full_dimension_reranks_only_low_dimension_candidates(self) -> None:
        query = full_vector(1.0, 0.0)
        query[64] = 2.0
        doc_a = full_vector(1.0, 0.0)
        doc_a[64] = -2.0
        doc_b = full_vector(0.8, 0.6)
        doc_b[64] = -2.0
        doc_c = full_vector(0.0, 1.0)
        doc_c[64] = 2.0
        corpus = [
            {"id": 1, "text": "doc-a"},
            {"id": 2, "text": "doc-b"},
            {"id": 3, "text": "doc-c"},
        ]
        model = FakeModel({"doc-a": doc_a, "doc-b": doc_b, "doc-c": doc_c, "query": query})
        searcher = ArabicSearcher(corpus, model=model)

        cascade = searcher.retrieve("query", method="cascade", dim=64, candidate_k=2, top_k=2)
        full = searcher.retrieve("query", method="semantic", dim=1024, top_k=3)

        self.assertEqual([hit["id"] for hit in cascade], [1, 2])
        self.assertEqual([hit["id"] for hit in full], [3, 1, 2])
        self.assertTrue(all(hit["score_kind"] == "cosine" for hit in cascade))


class PassageEmbeddingCacheTests(unittest.TestCase):
    @staticmethod
    def caching_searcher(
        corpus: list[dict], model: FakeModel, directory: str, *, revision: str = "test-revision"
    ) -> ArabicSearcher:
        searcher = ArabicSearcher(
            corpus,
            model=model,
            model_name="test-model",
            model_revision=revision,
            cache_dir=directory,
            load_embeddings=False,
        )
        # Injected models intentionally skip disk writes in normal use. Enable only
        # the cache path so this test remains isolated from model downloading.
        searcher._injected_model = False
        return searcher

    def test_reordered_and_expanded_corpus_encodes_only_missing_unique_text(self) -> None:
        vectors = {
            "doc-a": full_vector(1.0, 0.0),
            "doc-b": full_vector(0.0, 1.0),
            "doc-c": full_vector(1.0, 1.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            initial_model = FakeModel(vectors)
            initial = self.caching_searcher(
                [{"id": 1, "text": "doc-a"}, {"id": 2, "text": "doc-b"}],
                initial_model,
                directory,
            )
            initial_embeddings = initial.embeddings.copy()

            expanded_model = FakeModel(vectors)
            expanded = self.caching_searcher(
                [
                    {"id": 2, "text": "doc-b"},
                    {"id": 1, "text": "doc-a"},
                    {"id": 3, "text": "doc-c"},
                    {"id": 4, "text": "doc-a"},
                ],
                expanded_model,
                directory,
            )
            expanded_embeddings = expanded.embeddings

            self.assertEqual(initial_model.encoded_inputs, [["doc-a", "doc-b"]])
            self.assertEqual(expanded_model.encoded_inputs, [["doc-c"]])
            np.testing.assert_allclose(expanded_embeddings[0], initial_embeddings[1])
            np.testing.assert_allclose(expanded_embeddings[1], initial_embeddings[0])
            np.testing.assert_allclose(expanded_embeddings[1], expanded_embeddings[3])
            self.assertEqual(len(list(Path(directory).glob("*.npy"))), 3)

    def test_malformed_cache_entry_is_reencoded_and_repaired(self) -> None:
        corpus = [{"id": 1, "text": "doc-a"}]
        vectors = {"doc-a": full_vector(3.0, 4.0)}
        with tempfile.TemporaryDirectory() as directory:
            first = self.caching_searcher(corpus, FakeModel(vectors), directory)
            cache_path = first._cache_path("doc-a")
            self.assertIsNotNone(cache_path)
            _ = first.embeddings
            cache_path.write_bytes(b"not a numpy file")  # type: ignore[union-attr]

            replacement_model = FakeModel(vectors)
            reopened = self.caching_searcher(corpus, replacement_model, directory)
            repaired = reopened.embeddings

            self.assertEqual(replacement_model.encoded_inputs, [["doc-a"]])
            np.testing.assert_allclose(np.linalg.norm(repaired, axis=1), [1.0])
            cached = np.load(cache_path, allow_pickle=False)  # type: ignore[arg-type]
            self.assertEqual(cached.shape, (DEFAULT_DIM,))

    def test_cache_identity_includes_model_revision(self) -> None:
        corpus = [{"id": 1, "text": "doc-a"}]
        vectors = {"doc-a": full_vector(1.0)}
        with tempfile.TemporaryDirectory() as directory:
            first = self.caching_searcher(corpus, FakeModel(vectors), directory, revision="revision-a")
            _ = first.embeddings
            second_model = FakeModel(vectors)
            second = self.caching_searcher(corpus, second_model, directory, revision="revision-b")
            _ = second.embeddings

            self.assertNotEqual(first._cache_path("doc-a"), second._cache_path("doc-a"))
            self.assertEqual(second_model.encoded_inputs, [["doc-a"]])
            self.assertEqual(len(list(Path(directory).glob("*.npy"))), 2)

    def test_injected_model_does_not_write_disk_cache_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel({"doc": full_vector(1.0)})
            ArabicSearcher([{"id": 1, "text": "doc"}], model=model, cache_dir=directory)

            self.assertEqual(list(Path(directory).iterdir()), [])


class CorpusInvariantTests(unittest.TestCase):
    def test_validate_corpus_rejects_invalid_rows(self) -> None:
        invalid_corpora = [
            "not-a-list",
            [None],
            [{"id": True, "text": "valid"}],
            [{"id": 1, "text": "valid"}, {"id": 1, "text": "also valid"}],
            [{"id": 1, "text": "   "}],
            [{"id": 1}],
        ]
        for corpus in invalid_corpora:
            with self.subTest(corpus=corpus):
                with self.assertRaises(ValueError):
                    validate_corpus(corpus)  # type: ignore[arg-type]

    def test_empty_corpus_is_searchable_without_loading_a_model(self) -> None:
        searcher = ArabicSearcher([], load_embeddings=False)

        for method in ("semantic", "keyword", "hybrid", "cascade"):
            with self.subTest(method=method):
                self.assertEqual(searcher.retrieve("query", method=method), [])
        self.assertEqual(searcher.embeddings.shape, (0, DEFAULT_DIM))
        self.assertIsNone(searcher._model)

    def test_load_corpus_applies_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('[{"id": 1, "text": ""}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty text"):
                load_corpus(path)

    def test_builtin_corpus_has_stable_ids_categories_and_text(self) -> None:
        corpus = load_corpus()
        ids = [doc["id"] for doc in corpus]
        categories = {doc["category"] for doc in corpus}
        expected_categories = {
            "culture",
            "economy",
            "education",
            "health",
            "poetry",
            "religion",
            "science",
            "sports",
            "tech",
        }

        self.assertEqual(sorted(ids), list(range(1, len(corpus) + 1)))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(categories, expected_categories)
        self.assertTrue(all(doc["text"].strip() for doc in corpus))


if __name__ == "__main__":
    unittest.main()
