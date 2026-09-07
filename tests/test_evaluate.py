from __future__ import annotations

import math
import unittest
from pathlib import Path

from evaluate import (
    LEGACY_REGRESSION_SET,
    EvalCase,
    _embedding_memory,
    _review_status,
    bootstrap_comparisons,
    evaluate_cases,
    evaluate_dim,
    grouped_metrics,
    legacy_cases,
    load_cases_jsonl,
    load_corpus_json,
    ndcg_at_k,
    paired_bootstrap_difference,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_evaluation,
    success_at_k,
    summarize_no_answer,
    summarize_results,
    validate_cases,
)


FIXTURES = Path(__file__).parent / "fixtures"


class MetricTests(unittest.TestCase):
    def test_precision_at_k_uses_requested_depth(self) -> None:
        self.assertAlmostEqual(precision_at_k([1, 2, 3], frozenset({1, 3}), 3), 2 / 3)

    def test_reciprocal_rank_is_explicitly_cut_off_at_five(self) -> None:
        self.assertAlmostEqual(reciprocal_rank([2, 4, 3], frozenset({3})), 1 / 3)
        self.assertEqual(reciprocal_rank([2, 4, 5, 6, 7, 3], frozenset({3})), 0.0)

    def test_recall_and_success_have_distinct_meanings(self) -> None:
        retrieved = [1, 9, 2, 8, 7]
        relevant = frozenset({1, 2, 3})
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 5), 2 / 3)
        self.assertEqual(success_at_k(retrieved, relevant, 3), 1.0)

    def test_ndcg_at_k_uses_binary_relevance_and_ideal_dcg(self) -> None:
        actual = ndcg_at_k([2, 3, 1], frozenset({1, 3}), 3)
        dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(4))
        idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
        self.assertAlmostEqual(actual, dcg / idcg)

    def test_answer_metrics_reject_empty_relevance(self) -> None:
        with self.assertRaisesRegex(ValueError, "undefined"):
            recall_at_k([1], frozenset(), 5)
        with self.assertRaisesRegex(ValueError, "undefined"):
            ndcg_at_k([1], frozenset(), 5)


class StubSearcher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def search_by_vector(self, query_vector: list[int], top_k: int, dim: int) -> list[dict]:
        self.calls.append((top_k, dim))
        return [{"id": doc_id} for doc_id in query_vector[:top_k]]


class EvaluateDimTests(unittest.TestCase):
    def test_evaluate_dim_uses_preencoded_queries(self) -> None:
        searcher = StubSearcher()
        first = EvalCase("first", frozenset({1}), "unit", "test", "a")
        second = EvalCase("second", frozenset({9}), "unit", "test", "b")
        encoded_queries = [([2, 1, 3], first), ([4, 5, 6], second)]

        metrics = evaluate_dim(searcher, 64, encoded_queries)

        self.assertEqual(searcher.calls, [(5, 64), (5, 64)])
        self.assertAlmostEqual(metrics["MRR@5"], 0.25)
        self.assertAlmostEqual(metrics["P@1"], 0.0)
        self.assertAlmostEqual(metrics["P@3"], 1 / 6)
        self.assertAlmostEqual(metrics["Success@5"], 0.5)

    def test_grouped_metrics_reports_each_bucket(self) -> None:
        searcher = StubSearcher()
        encoded_queries = [
            ([1, 2, 3], EvalCase("a", frozenset({1}), "easy", "health", "a")),
            ([2, 1, 3], EvalCase("b", frozenset({1}), "hard", "health", "b")),
            ([4, 5, 6], EvalCase("c", frozenset({9}), "hard", "tech", "c")),
        ]

        results = evaluate_cases(searcher, 1024, encoded_queries)
        overall = summarize_results(results)
        by_tier = grouped_metrics(results, lambda row: row.case.tier)
        by_category = grouped_metrics(results, lambda row: row.case.category)

        self.assertAlmostEqual(overall["P@1"], 1 / 3)
        self.assertEqual(by_tier["easy"]["n"], 1)
        self.assertEqual(by_tier["hard"]["n"], 2)
        self.assertEqual(by_category["health"]["n"], 2)
        self.assertEqual(by_category["tech"]["n"], 1)

    def test_no_answer_queries_do_not_pollute_answerable_metrics(self) -> None:
        searcher = StubSearcher()
        rows = [
            ([1], EvalCase("answer", frozenset({1}), "unit", "x", "a")),
            ([9], EvalCase("unknown", frozenset(), "no_answer", "x", "b")),
        ]
        results = evaluate_cases(searcher, 64, rows)

        metrics = summarize_results(results)
        no_answer = summarize_no_answer(results)

        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["excluded_no_answer"], 1)
        self.assertEqual(metrics["MRR@5"], 1.0)
        self.assertEqual(no_answer["n"], 1)
        self.assertEqual(no_answer["returned_any_rate"], 1.0)


class KeywordOnlySearcher:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.retrieve_calls = 0

    def encode_queries(self, queries: list[str]) -> list[list[int]]:
        self.encode_calls += 1
        raise AssertionError("keyword evaluation must not encode queries")

    def retrieve(self, query: str, **kwargs: object) -> list[dict]:
        self.retrieve_calls += 1
        return [{"id": 1}]


class PipelineTests(unittest.TestCase):
    def test_keyword_method_never_encodes(self) -> None:
        searcher = KeywordOnlySearcher()
        cases = [EvalCase("سؤال", frozenset({1}), "unit", "test", "q1", "test")]

        evidence, timing = run_evaluation(
            searcher,
            cases,
            dimensions=[64],
            methods=["keyword"],
            repeats=3,
            seed=7,
            bootstrap_resamples=20,
        )

        self.assertEqual(searcher.encode_calls, 0)
        self.assertEqual(searcher.retrieve_calls, 4)
        self.assertIsNone(timing["query_encoding_seconds"])
        self.assertEqual(
            evidence["results"]["keyword"]["64"]["answerable"]["MRR@5"],
            1.0,
        )
        timing_result = evidence["results"]["keyword"]["64"]["timing"]
        self.assertEqual(timing_result["warm_repetitions"]["repeats"], 3)
        self.assertEqual(timing_result["warmup_runs_including_cold"], 1)

    def test_bootstrap_is_paired_deterministic_and_reports_groups(self) -> None:
        searcher = StubSearcher()
        cases = [
            EvalCase("a", frozenset({1}), "easy", "health", "a"),
            EvalCase("b", frozenset({1}), "hard", "health", "b"),
        ]
        candidate = evaluate_cases(searcher, 64, [([1], cases[0]), ([2, 1], cases[1])])
        reference = evaluate_cases(searcher, 1024, [([1], cases[0]), ([1], cases[1])])

        first = paired_bootstrap_difference(
            candidate,
            reference,
            lambda row: row.reciprocal_rank,
            resamples=100,
            seed=11,
        )
        second = paired_bootstrap_difference(
            candidate,
            reference,
            lambda row: row.reciprocal_rank,
            resamples=100,
            seed=11,
        )
        grouped = bootstrap_comparisons(candidate, reference, resamples=20, seed=3)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["candidate_minus_1024"], -0.25)
        self.assertIn("easy", grouped["by_tier"])
        self.assertIn("health", grouped["by_category"])
        self.assertIn("Shared relevant documents", grouped["caveat"])


class InputValidationTests(unittest.TestCase):
    def test_synthetic_fixture_is_valid_and_explicitly_unreviewed(self) -> None:
        corpus = load_corpus_json(FIXTURES / "eval_synthetic_corpus.json")
        cases, metadata = load_cases_jsonl(FIXTURES / "eval_synthetic_cases.jsonl")
        validate_cases(cases, {doc["id"] for doc in corpus})

        self.assertGreaterEqual(len(corpus), 15)
        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual(metadata["human_review_status"], "UNREVIEWED")
        self.assertTrue(metadata["synthetic"])
        self.assertEqual({row.split for row in cases}, {"development", "calibration", "test"})
        self.assertGreaterEqual(sum(not row.relevant for row in cases), 3)

    def test_human_review_claim_requires_supporting_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires reviewer"):
            _review_status({"human_review": {"status": "HUMAN_REVIEWED"}})
        self.assertEqual(
            _review_status(
                {
                    "human_review": {
                        "status": "HUMAN_REVIEWED",
                        "reviewer": "review-team",
                        "reviewed_at": "2026-09-07",
                        "protocol": "dual relevance review with adjudication",
                    }
                }
            ),
            "HUMAN_REVIEWED",
        )

    def test_embedding_memory_does_not_double_count_1024_alias(self) -> None:
        array = type("Array", (), {"nbytes": 4096})()
        searcher = type("Searcher", (), {})()
        searcher._embeddings = array
        searcher._embeddings_by_dim = {1024: array}
        memory = _embedding_memory(searcher)
        self.assertEqual(memory["full_embedding_matrix_bytes"], 4096)
        self.assertEqual(memory["additional_unique_per_dimension_cache_bytes"], 0)
        self.assertEqual(memory["known_embedding_arrays_total_bytes"], 4096)

    def test_duplicate_normalized_queries_are_rejected(self) -> None:
        cases = [
            EvalCase("أهلاً   بالعالم", frozenset({1}), "x", "x", "a", "test"),
            EvalCase("أهلاً بالعالم", frozenset({1}), "x", "x", "b", "test"),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate normalized query"):
            validate_cases(cases, {1})

    def test_missing_relevant_document_is_rejected(self) -> None:
        cases = [EvalCase("query", frozenset({999}), "x", "x", "a", "test")]
        with self.assertRaisesRegex(ValueError, "absent corpus ids"):
            validate_cases(cases, {1})


class LegacyRegressionTests(unittest.TestCase):
    def test_legacy_benchmark_remains_exactly_120_regression_queries(self) -> None:
        self.assertEqual(len(LEGACY_REGRESSION_SET), 120)
        self.assertTrue(all(row.split == "regression" for row in LEGACY_REGRESSION_SET))
        self.assertEqual(len(legacy_cases()), 120)

    def test_legacy_cases_have_metadata(self) -> None:
        self.assertTrue(all(row.query.strip() for row in LEGACY_REGRESSION_SET))
        self.assertTrue(all(row.relevant for row in LEGACY_REGRESSION_SET))
        self.assertTrue(all(row.tier.strip() for row in LEGACY_REGRESSION_SET))
        self.assertTrue(all(row.category.strip() for row in LEGACY_REGRESSION_SET))


if __name__ == "__main__":
    unittest.main()
