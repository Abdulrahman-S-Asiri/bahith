# Retrieval evaluation

`evaluate.py` produces machine-readable JSON evidence for Bahith retrieval runs. The embedded 120-query suite is preserved for regression detection only. Its queries were written for the 30-document demo corpus, so its scores are not independent evidence of production Arabic-search quality.

The small files under `tests/fixtures/` are synthetic, unreviewed test fixtures. They exercise Modern Standard Arabic, Saudi colloquial phrasing, named entities, and no-answer handling. They are useful for testing the evaluator and are not a benchmark claim.

## Run an evaluation

Run the embedded regression suite across Matryoshka dimensions:

```powershell
python evaluate.py --output outputs/regression.json --dimensions 64,128,256,512,768,1024 --methods semantic --warmup-runs 1 --repeats 5 --seed 20260907
```

Run externally supplied cases and compare retrieval methods:

```powershell
python evaluate.py --corpus path/to/corpus.json --cases path/to/cases.jsonl --output outputs/evaluation.json --dimensions 256,1024 --methods semantic,keyword,hybrid,cascade --warmup-runs 1 --repeats 5 --seed 20260907
```

Use `--splits test` to report a held-out split. Development and calibration queries should be used to choose methods, weights, thresholds, or prompts; the test split should remain untouched until those choices are fixed.

The JSON records corpus, case, evaluator-source, and search-source SHA-256 hashes; model name and revision; query prompt; Python and dependency versions; hardware information; run configuration; aggregate metrics; and every per-query retrieval result. Source hashes identify the executed code even when the working tree is ahead of Git `HEAD`. The seed initializes available Python, NumPy, and Torch random generators and makes paired bootstrap resampling reproducible. Exact ranking can still vary if the selected numerical backend is nondeterministic.

## Case format

Cases are UTF-8 JSON Lines. An optional first metadata record describes provenance and review:

```json
{"_meta":{"name":"arabic-search-v1","human_review":{"status":"UNREVIEWED"}}}
{"id":"q001","query":"كيف أجدد جواز السفر؟","relevant_ids":[17],"tier":"paraphrase","category":"public_services","split":"test"}
{"id":"q002","query":"ما سعر التذكرة؟","relevant_ids":[],"no_answer":true,"tier":"no_answer","category":"transport","split":"test"}
```

Each case needs a unique `id`, nonempty `query`, `tier`, `category`, and one of the split labels `development`, `calibration`, or `test`. Every relevant ID and optional `document_id` must exist in the corpus. Duplicate IDs and Unicode-normalized duplicate queries are rejected. An answerable query needs at least one `relevant_ids` entry. A no-answer query needs an empty list and `no_answer: true`.

Evaluation evidence reports `UNREVIEWED` unless the metadata explicitly says `HUMAN_REVIEWED` and includes nonempty `reviewer`, `reviewed_at`, and `protocol` fields. This records the supplied review claim; it cannot independently verify that a review occurred.

## Metrics

Answerable-query metrics are binary-relevance metrics at a fixed retrieval depth of five:

- `MRR@5`: reciprocal rank of the first relevant document, with misses after rank five scored as zero.
- `NDCG@5`: discounted gain normalized by the ideal placement of all relevant documents available within five slots.
- `Recall@5`: fraction of all labeled relevant documents retrieved in the first five.
- `Success@3` and `Success@5`: fraction of queries with at least one relevant result in the first three or five.
- `P@1` and `P@3`: relevant results divided by the requested cutoff.

No-answer queries are excluded from every answerable metric. They receive a separate abstention/returned-results summary. Because Bahith generally returns ranked results without a calibrated abstention threshold, a high returned-results rate on these queries is expected and should not be disguised as answerable-query failure.

For every evaluated dimension below 1024, the report includes paired query-bootstrap 95% intervals for the difference from 1024 dimensions, both overall and by tier/category. Positive values favor the lower-dimensional candidate. Queries sharing relevant documents can have correlated errors, violating the independent-query resampling assumption and making intervals too narrow; the JSON repeats this caveat. Small groups also produce unstable intervals.

## Timing and memory interpretation

Corpus loading, searcher/model boot, and semantic query encoding are timed separately. Semantic ranking repetition excludes query encoding. Keyword timing measures retrieval without model encoding. Hybrid and cascade timing covers their complete public `retrieve()` pipeline, including any query encoding it performs. Quality metrics never contain timing values.

Each configuration clears the query cache, reports its first full batch separately as `cache_cold_batch_seconds`, performs the requested warm-up count, and then records median and p95 wall-clock time across warm repetitions. Those percentiles describe complete query-set batches, not per-query latency; query count and median throughput are included. This is cold with respect to the query cache; model and corpus boot are reported separately. Cold and warm samples are never pooled. These are local measurements, not portable latency guarantees. Use a controlled process and hardware for formal performance comparisons.

The evidence records known embedding-array payload bytes. A shared process retains the full 1024-dimensional matrix and any dimension-specific caches, so selecting a smaller dimension in that process does not prove lower process memory. A defensible memory comparison requires separate processes that load only the tested representation, plus process-level peak-memory measurement.
