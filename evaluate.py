"""Measure retrieval accuracy of بَاحث across all Matryoshka dimensions.

Runs a hand-crafted hard test set (paraphrase / oblique / adversarial)
against the project's own ArabicSearcher at every supported truncation
dimension and prints MRR, NDCG@5, P@1, P@3 per dim.

Usage:  python evaluate.py
"""
from __future__ import annotations

import math
import time

from search import ArabicSearcher, SUPPORTED_DIMS, load_corpus

# ---------------------------------------------------------------------------
# Test set: (query, frozenset of relevant doc ids)
# Strict binary relevance — only docs that *directly* answer the query.
# ---------------------------------------------------------------------------

# Tier 1 · Paraphrase: little or no surface-word overlap with the target.
PARAPHRASE: list[tuple[str, frozenset[int]]] = [
    ("آلاف الناس فقدوا أعمالهم في القطاع التقني",                       frozenset({23})),
    ("مهارات الحاسب في فهم لغات البشر والتعامل معها آلياً",            frozenset({14})),
    ("عملة لا تخضع لسلطة مصرفية مركزية",                                frozenset({12})),
    ("كيف يحسّن الإنسان من إدراكه ووعيه؟",                               frozenset({27})),
    ("أخبار من علم الفلك حول وجود الماء على أجرام بعيدة",              frozenset({16})),
    ("نص شرعي يربط الجزاء بالقصد دون الفعل الظاهر",                     frozenset({4})),
    ("منافسات دولية وانكسار توقيت قياسي",                               frozenset({26})),
    ("كيف يقاوم الجسم الجراثيم باستخدام تكنولوجيا حديثة؟",             frozenset({17})),
    ("حكم في تشجيع الكدّ والمثابرة لطلب الأرزاق",                        frozenset({1, 18})),
    ("أنواع الأنشطة التي تنفع جهاز الدوران",                            frozenset({6})),
    ("لماذا يحبّ الأطباء أن يضمّ الإنسان الفاكهة إلى وجباته؟",          frozenset({9})),
    ("ما الذي تركه أهل المخا للعالم منذ مئات السنين؟",                 frozenset({30})),
]

# Tier 2 · Oblique: query asks about an implication, not the literal claim.
OBLIQUE: list[tuple[str, frozenset[int]]] = [
    ("هل سهر الليل يضرّ الطلبة في الامتحانات؟",                          frozenset({8})),
    ("كيف وصل الإنسان إلى سطح الكوكب الأحمر مؤخرًا؟",                   frozenset({15})),
    ("هل خدمة المجتمع التطوعية مهمة؟",                                   frozenset({5})),
    ("ما أثر السياسة النقدية الحكومية على حركة المستثمرين؟",            frozenset({24})),
    ("ما الذي يجعل الترجمة الآلية في وقتنا أكثر دقة من قبل؟",           frozenset({14})),
    ("أيهما أنفع للذاكرة: ساعات راحة كافية أم تدريبات ذهنية؟",          frozenset({8})),
    ("ماذا يقول التراث العربي عن من أحسن إلى من لا يستحق؟",            frozenset({21})),
    ("ما الذي يدفع الدول المنتجة للنفط إلى تنسيق إنتاجها؟",             frozenset({22})),
    ("كيف نتعرّف على أصحاب الإرادة الحقيقية في الحياة؟",                frozenset({18, 20})),
    ("هل غيّرت الجائحة طريقة الدراسة في المدارس؟",                       frozenset({28})),
]

# Tier 3 · Adversarial: lexical overlap with WRONG docs; meaning must win.
ADVERSARIAL: list[tuple[str, frozenset[int]]] = [
    ("ما أهمية ممارسة الرياضة لصحة الإنسان البالغ؟",                    frozenset({6})),
    ("ما العلاقة بين النية والإخلاص في العمل في الإسلام؟",              frozenset({4})),
    ("ما الذي يميز اليمنيين في تجارة منتجاتهم الزراعية؟",              frozenset({30})),
    ("كيف يفهم الذكاء الاصطناعي جملةً عربيةً معقدة؟",                  frozenset({14})),
    ("كيف نضمن سلامة التحويلات الرقمية بين الأطراف؟",                   frozenset({12})),
    ("نصائح غذائية للوقاية من أمراض القلب والشرايين",                  frozenset({9})),
    ("بيت شعري عن أن الكبار وحدهم يستطيعون الإنجازات الكبيرة",        frozenset({20})),
    ("ما الذي يميّز التعليم بعد عام 2020؟",                              frozenset({28})),
]

TEST_SET = PARAPHRASE + OBLIQUE + ADVERSARIAL
K = 5  # retrieval depth used for all metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[int], relevant: frozenset[int], k: int) -> float:
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def reciprocal_rank(retrieved: list[int], relevant: frozenset[int]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: frozenset[int], k: int) -> float:
    # Binary relevance, so DCG numerators are 0 or 1.
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, d in enumerate(retrieved[:k], 1)
        if d in relevant
    )
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-dimension evaluation
# ---------------------------------------------------------------------------

def evaluate_dim(searcher: ArabicSearcher, dim: int) -> dict[str, float]:
    mrr = ndcg = p1 = p3 = 0.0
    for query, relevant in TEST_SET:
        results = searcher.search(query, top_k=K, dim=dim)
        retrieved = [r["id"] for r in results]
        mrr  += reciprocal_rank(retrieved, relevant)
        ndcg += ndcg_at_k(retrieved, relevant, K)
        p1   += precision_at_k(retrieved, relevant, 1)
        p3   += precision_at_k(retrieved, relevant, 3)
    n = len(TEST_SET)
    return {"MRR": mrr / n, "NDCG@5": ndcg / n, "P@1": p1 / n, "P@3": p3 / n}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading model and corpus...")
    t0 = time.perf_counter()
    corpus = load_corpus()
    searcher = ArabicSearcher(corpus)
    boot = time.perf_counter() - t0
    print(f"Ready in {boot:.1f}s · {len(corpus)} docs · {len(TEST_SET)} queries\n")

    print("Evaluating across Matryoshka dimensions...\n")
    rows = []
    for dim in sorted(SUPPORTED_DIMS, reverse=True):
        t = time.perf_counter()
        metrics = evaluate_dim(searcher, dim)
        rows.append((dim, metrics, time.perf_counter() - t))

    width = 64
    print("=" * width)
    print(f"  {'Dim':>5}    {'MRR':>6}   {'NDCG@5':>7}    {'P@1':>5}    {'P@3':>5}    {'Time':>7}")
    print("-" * width)
    for dim, m, t in rows:
        print(
            f"  {dim:>5}    {m['MRR']:>6.3f}   {m['NDCG@5']:>7.3f}    "
            f"{m['P@1']:>5.3f}    {m['P@3']:>5.3f}    {t:>5.2f}s"
        )
    print("=" * width)
    print(
        f"n={len(TEST_SET)} queries "
        f"({len(PARAPHRASE)} paraphrase + {len(OBLIQUE)} oblique + "
        f"{len(ADVERSARIAL)} adversarial) · k={K} · binary relevance"
    )


if __name__ == "__main__":
    main()
