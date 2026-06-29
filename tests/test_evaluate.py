from __future__ import annotations

import math
import unittest

from evaluate import (
    TEST_SET,
    EvalCase,
    evaluate_cases,
    evaluate_dim,
    grouped_metrics,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    summarize_results,
)


class MetricTests(unittest.TestCase):
    def test_precision_at_k_uses_requested_depth(self) -> None:
        self.assertAlmostEqual(precision_at_k([1, 2, 3], frozenset({1, 3}), 3), 2 / 3)

    def test_reciprocal_rank_returns_first_relevant_hit(self) -> None:
        self.assertAlmostEqual(reciprocal_rank([2, 4, 3], frozenset({3})), 1 / 3)
        self.assertEqual(reciprocal_rank([2, 4, 5], frozenset({3})), 0.0)

    def test_ndcg_at_k_uses_binary_relevance_and_ideal_dcg(self) -> None:
        actual = ndcg_at_k([2, 3, 1], frozenset({1, 3}), 3)
        dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(4))
        idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))

        self.assertAlmostEqual(actual, dcg / idcg)


class StubSearcher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def search_by_vector(self, query_vector: list[int], top_k: int, dim: int) -> list[dict]:
        self.calls.append((top_k, dim))
        return [{"id": doc_id} for doc_id in query_vector[:top_k]]


class EvaluateDimTests(unittest.TestCase):
    def test_evaluate_dim_uses_preencoded_queries(self) -> None:
        searcher = StubSearcher()
        first = EvalCase("first", frozenset({1}), "unit", "test")
        second = EvalCase("second", frozenset({9}), "unit", "test")
        encoded_queries = [
            ([2, 1, 3], first),
            ([4, 5, 6], second),
        ]

        metrics = evaluate_dim(searcher, 64, encoded_queries)

        self.assertEqual(searcher.calls, [(5, 64), (5, 64)])
        self.assertAlmostEqual(metrics["MRR"], 0.25)
        self.assertAlmostEqual(metrics["P@1"], 0.0)
        self.assertAlmostEqual(metrics["P@3"], 1 / 6)

    def test_grouped_metrics_reports_each_bucket(self) -> None:
        searcher = StubSearcher()
        encoded_queries = [
            ([1, 2, 3], EvalCase("a", frozenset({1}), "easy", "health")),
            ([2, 1, 3], EvalCase("b", frozenset({1}), "hard", "health")),
            ([4, 5, 6], EvalCase("c", frozenset({9}), "hard", "tech")),
        ]

        results = evaluate_cases(searcher, 1024, encoded_queries)
        overall = summarize_results(results)
        by_tier = grouped_metrics(results, lambda row: row.case.tier)
        by_category = grouped_metrics(results, lambda row: row.case.category)

        self.assertAlmostEqual(overall["P@1"], 1 / 3)
        self.assertEqual(by_tier["easy"]["n"], 1.0)
        self.assertEqual(by_tier["hard"]["n"], 2.0)
        self.assertEqual(by_category["health"]["n"], 2.0)
        self.assertEqual(by_category["tech"]["n"], 1.0)


class TestSetCoverageTests(unittest.TestCase):
    def test_benchmark_has_production_confidence_query_count(self) -> None:
        self.assertGreaterEqual(len(TEST_SET), 100)

    def test_benchmark_cases_have_metadata(self) -> None:
        self.assertTrue(all(row.query.strip() for row in TEST_SET))
        self.assertTrue(all(row.relevant for row in TEST_SET))
        self.assertTrue(all(row.tier.strip() for row in TEST_SET))
        self.assertTrue(all(row.category.strip() for row in TEST_SET))


if __name__ == "__main__":
    unittest.main()
