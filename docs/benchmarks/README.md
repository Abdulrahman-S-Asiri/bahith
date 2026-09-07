# Bahith retrieval evidence

These results verify retrieval behavior on Bahith's fixed demo corpus. They are regression evidence, not an independent Arabic-search benchmark. The 120 queries were hand-authored against 30 known passages, have not received documented human review, and contain no no-answer cases.

Raw evidence:

- [Matryoshka dimension regression](matryoshka-regression.json)
- [Retrieval method comparison](method-comparison.json)
- [Synthetic pipeline check](synthetic-pipeline-check.json) — separate, unreviewed fixture evidence.

Both runs used `Omartificial-Intelligence-Space/Harrier-Arabic-Matryoshka-0.6B` at revision `397a969cf4db4b9bcda254e857823d87077b7e6f`, with an empty query prompt, normalized embeddings, five warm timing repetitions, and seed `20260907`. All repeated rankings were consistent.

## Matryoshka dimensions

All metrics use binary relevance at a retrieval depth of five. `MRR@5` stops after rank five. The confidence interval is the paired query-bootstrap difference in MRR@5 from 1024 dimensions; negative values favor 1024.

| Dimensions | P@1 | MRR@5 | NDCG@5 | Recall@5 | Success@5 | Δ MRR@5 vs 1024 (95% CI) | Warm median, 120-query batch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | .7083 | .7871 | .8180 | .9083 | .9083 | −.1410 [−.1918, −.0922] | 8.20 ms |
| 128 | .8000 | .8554 | .8739 | .9292 | .9333 | −.0726 [−.1129, −.0365] | 8.15 ms |
| 256 | .8417 | .8875 | .9020 | .9458 | .9500 | −.0406 [−.0753, −.0104] | 8.18 ms |
| 512 | .8833 | .9197 | .9297 | .9625 | .9667 | −.0083 [−.0236, +.0031] | 8.69 ms |
| 768 | .8917 | .9194 | .9273 | .9542 | .9583 | −.0086 [−.0197, −.0014] | 8.70 ms |
| 1024 | **.9000** | **.9281** | **.9346** | .9583 | **.9667** | reference | 8.93 ms |

The exact 1024-dimensional P@1 is **.9000**, matching the original recorded run. This is the canonical reproduced value; historical rounded README figures should not replace it.

The suite shows clear ranking degradation at 64, 128, and 256 dimensions. At 512 dimensions the observed MRR@5 loss is 0.83 percentage points and its query-bootstrap interval includes zero. This means the run did not resolve a difference; it does not prove equivalence. The 768-dimensional MRR@5 loss is similarly small. The 1024-dimensional configuration remains the evidence-backed default because it has the best MRR@5, NDCG@5, and P@1, while this run establishes no material end-to-end performance benefit from truncation.

## Retrieval methods at 1024 dimensions

The method run used `candidate_k=10`. Cascade at 1024 dimensions is a consistency control: its candidate and reranking stages use the same full representation. It is not a compression result.

| Method | P@1 | MRR@5 | NDCG@5 | Recall@5 | Success@5 | Warm median, 120-query batch |
|---|---:|---:|---:|---:|---:|---:|
| Semantic | **.9000** | **.9281** | **.9346** | .9583 | **.9667** | 9.27 ms |
| Keyword | .6750 | .6919 | .6969 | .7125 | .7167 | 14.38 ms |
| Hybrid | .7583 | .8346 | .8635 | .9542 | .9583 | 31.71 ms |
| Cascade, 1024 control | **.9000** | **.9281** | **.9346** | .9583 | **.9667** | 19.43 ms |

Hybrid fusion is negative evidence in this setup. Against semantic retrieval, it improved MRR@5 on 3 queries, worsened it on 23, and left 94 unchanged; mean MRR@5 fell by 9.35 percentage points. Its recall stayed close because it often retained the relevant passage lower in the result list, but top-rank quality fell. Keyword retrieval was worse still. The default therefore remains semantic retrieval at 1024 dimensions. Fusion weights or candidate depth should not be tuned against this regression suite and then reported on the same queries.

## Timing scope

Timing separates model boot, semantic query encoding, and ranking. In the dimension run, boot took 59.68 seconds and one batched encoding of all 120 queries took 65.34 seconds. The 8.15–8.93 ms figures above measure ranking of the complete pre-encoded query set; they are not per-query end-to-end latency.

In the method run, boot took 34.60 seconds and batched semantic encoding took 71.36 seconds. Semantic warm timing reuses that pre-encoded batch. Keyword does not encode. Hybrid and cascade accept query text through the public retrieval API: their cache-cold 120-query passes took 69.92 and 70.58 seconds because they encoded queries individually, while their warm figures reused the query cache. Cold and warm samples are not pooled.

Only five warm repetitions were collected on one CPU environment and a 30-passage corpus. Sub-millisecond differences between dimensions are not portable performance evidence. Reported memory covers known array payloads in a shared full-vector process; it is not process peak memory and does not show that selecting a lower dimension reduces deployed RAM.

## Evidence limits and next gate

A separate run exercised 24 synthetic queries against 16 authored passages at 64, 256, and
1024 dimensions, across all four retrieval methods (`candidate_k=10`, three warm repetitions).
Twenty queries were answerable and four intentionally had no answer. The 64-dimensional
cascade reached MRR@5 1.000 on the 20 answerable fixture cases versus .9167 for 64-dimensional
semantic retrieval. This checks that the shortlist/full-vector path works on this fixture;
it is not an independently measured accuracy improvement. All methods returned results for
all four no-answer queries. That exposes the remaining absence of a calibrated abstention
policy rather than disguising it as successful answer finding. The synthetic data's split
labels exercise the file format; this all-splits check is not a held-out benchmark.

Ninety of the 120 queries are coverage variants, with several queries sharing each relevant passage. The paired bootstrap resamples queries as if they were independent, so its intervals can be too narrow. Tier and category groups are smaller still; one category contains a single query. These results support regression detection only. They do not establish human-perceived quality, production reliability, state of the art performance, general Arabic coverage, Saudi-dialect quality, no-answer behavior, or deployment speed.

The next acceptance gate is a frozen corpus that was not used to author or tune the current system, with realistic multi-passage documents and a preregistered development/calibration/test split. At least two fluent Arabic reviewers should independently label relevance and no-answer cases under a written protocol, then adjudicate disagreements. The set should deliberately cover Modern Standard Arabic, Saudi usage, names, lexical confounders, and document-scoped retrieval. Primary metrics, allowed degradation margins, subgroup checks, and end-to-end latency/memory procedures must be fixed before opening the test labels. Cluster-aware uncertainty by source document should accompany the query-level analysis.

The JSON files record SHA-256 hashes of the exact local source bytes used for execution: `evaluate.py` is `5de44f68a3a870b2e0e231ae83fd30b9946bc3970198fc79636aa1895ef33cd2` and `search.py` is `a766ef5d3576b31af0ad2cc37dfa5db035f810a6fe132c4a00ebe50e292586ed`. Because local line endings can differ from Git-normalized content, the public JSON copies also carry coordinator-generated normalized-LF hashes for repository verification. Public copies use sanitized corpus paths.
