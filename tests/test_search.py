from __future__ import annotations

import unittest

import numpy as np

from search import DEFAULT_DIM, ArabicSearcher, load_corpus


def full_vector(*values: float, dim: int = DEFAULT_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[: len(values)] = values
    return vec


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

    def test_truncate_rejects_invalid_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "dim must be one of"):
            ArabicSearcher._truncate(np.ones((1, DEFAULT_DIM), dtype=np.float32), 63)

    def test_model_dimension_is_validated(self) -> None:
        corpus = [{"id": 1, "category": "test", "text": "doc"}]
        model = FakeModel({"doc": full_vector(1.0, dim=768)})

        with self.assertRaisesRegex(ValueError, "expected 1024"):
            ArabicSearcher(corpus, model=model)

    def test_search_orders_by_score_then_corpus_order(self) -> None:
        searcher, _ = self.make_searcher()

        hits = searcher.search("query", top_k=3, dim=64)

        self.assertEqual([hit["id"] for hit in hits], [1, 2, 3])
        np.testing.assert_allclose([hit["score"] for hit in hits], [1.0, 1.0, 0.0])

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


class CorpusInvariantTests(unittest.TestCase):
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
