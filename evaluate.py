"""Measure retrieval accuracy of بَاحث across all Matryoshka dimensions.

Runs a hand-crafted Arabic benchmark against the project's ArabicSearcher
and reports overall, tier-level, category-level, and error-analysis metrics.

Usage:  python evaluate.py
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from search import ArabicSearcher, SUPPORTED_DIMS, load_corpus


@dataclass(frozen=True)
class EvalCase:
    query: str
    relevant: frozenset[int]
    tier: str
    category: str


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    retrieved: tuple[int, ...]
    reciprocal_rank: float
    ndcg_at_5: float
    p_at_1: float
    p_at_3: float


EncodedQuery = tuple[np.ndarray, EvalCase]
MetricRow = dict[str, float]
K = 5  # retrieval depth used for all metrics


def case(query: str, relevant: set[int], tier: str, category: str) -> EvalCase:
    return EvalCase(query, frozenset(relevant), tier, category)


# ---------------------------------------------------------------------------
# Test set: strict binary relevance.
# ---------------------------------------------------------------------------

# Tier 1 · Paraphrase: little or no surface-word overlap with the target.
PARAPHRASE: list[EvalCase] = [
    case("آلاف الناس فقدوا أعمالهم في القطاع التقني", {23}, "paraphrase", "economy"),
    case("مهارات الحاسب في فهم لغات البشر والتعامل معها آلياً", {14}, "paraphrase", "tech"),
    case("عملة لا تخضع لسلطة مصرفية مركزية", {12}, "paraphrase", "tech"),
    case("كيف يحسّن الإنسان من إدراكه ووعيه؟", {27}, "paraphrase", "education"),
    case("أخبار من علم الفلك حول وجود الماء على أجرام بعيدة", {16}, "paraphrase", "science"),
    case("نص شرعي يربط الجزاء بالقصد دون الفعل الظاهر", {4}, "paraphrase", "religion"),
    case("منافسات دولية وانكسار توقيت قياسي", {26}, "paraphrase", "sports"),
    case("كيف يقاوم الجسم الجراثيم باستخدام تكنولوجيا حديثة؟", {17}, "paraphrase", "science"),
    case("حكم في تشجيع الكدّ والمثابرة لطلب الأرزاق", {1, 18}, "paraphrase", "mixed"),
    case("أنواع الأنشطة التي تنفع جهاز الدوران", {6}, "paraphrase", "health"),
    case("لماذا يحبّ الأطباء أن يضمّ الإنسان الفاكهة إلى وجباته؟", {9}, "paraphrase", "health"),
    case("ما الذي تركه أهل المخا للعالم منذ مئات السنين؟", {30}, "paraphrase", "culture"),
]

# Tier 2 · Oblique: query asks about an implication, not the literal claim.
OBLIQUE: list[EvalCase] = [
    case("هل سهر الليل يضرّ الطلبة في الامتحانات؟", {8}, "oblique", "health"),
    case("كيف وصل الإنسان إلى سطح الكوكب الأحمر مؤخرًا؟", {15}, "oblique", "science"),
    case("هل خدمة المجتمع التطوعية مهمة؟", {5}, "oblique", "religion"),
    case("ما أثر السياسة النقدية الحكومية على حركة المستثمرين؟", {24}, "oblique", "economy"),
    case("ما الذي يجعل الترجمة الآلية في وقتنا أكثر دقة من قبل؟", {14}, "oblique", "tech"),
    case("أيهما أنفع للذاكرة: ساعات راحة كافية أم تدريبات ذهنية؟", {8}, "oblique", "health"),
    case("ماذا يقول التراث العربي عن من أحسن إلى من لا يستحق؟", {21}, "oblique", "poetry"),
    case("ما الذي يدفع الدول المنتجة للنفط إلى تنسيق إنتاجها؟", {22}, "oblique", "economy"),
    case("كيف نتعرّف على أصحاب الإرادة الحقيقية في الحياة؟", {18, 20}, "oblique", "poetry"),
    case("هل غيّرت الجائحة طريقة الدراسة في المدارس؟", {28}, "oblique", "education"),
]

# Tier 3 · Adversarial: lexical overlap with wrong docs; meaning must win.
ADVERSARIAL: list[EvalCase] = [
    case("ما أهمية ممارسة الرياضة لصحة الإنسان البالغ؟", {6}, "adversarial", "health"),
    case("ما العلاقة بين النية والإخلاص في العمل في الإسلام؟", {4}, "adversarial", "religion"),
    case("ما الذي يميز اليمنيين في تجارة منتجاتهم الزراعية؟", {30}, "adversarial", "culture"),
    case("كيف يفهم الذكاء الاصطناعي جملةً عربيةً معقدة؟", {14}, "adversarial", "tech"),
    case("كيف نضمن سلامة التحويلات الرقمية بين الأطراف؟", {12}, "adversarial", "tech"),
    case("نصائح غذائية للوقاية من أمراض القلب والشرايين", {9}, "adversarial", "health"),
    case("بيت شعري عن أن الكبار وحدهم يستطيعون الإنجازات الكبيرة", {20}, "adversarial", "poetry"),
    case("ما الذي يميّز التعليم بعد عام 2020؟", {28}, "adversarial", "education"),
]

# Tier 4 · Coverage: three extra labeled queries per corpus document.
COVERAGE: list[EvalCase] = [
    case("آية تمنح الأمل بعد الضيق وتدعو للعمل بعد الفراغ", {1}, "coverage", "religion"),
    case("ماذا أفعل عندما تنتهي مهمة صعبة وأريد التوجه إلى الله؟", {1}, "coverage", "religion"),
    case("معنى أن الفرج يأتي مع الصبر على الشدة", {1}, "coverage", "religion"),
    case("طلب الزيادة في العلم وربط المعرفة بالنور", {2}, "coverage", "religion"),
    case("دعاء قرآني لمن يريد التعلم والفهم", {2}, "coverage", "religion"),
    case("لماذا يمدح النص العلم ويحذر من الجهل؟", {2}, "coverage", "religion"),
    case("وعد ديني بالفرج والرزق لمن يتقي الله", {3}, "coverage", "religion"),
    case("كيف تفتح التقوى أبواب الحلول غير المتوقعة؟", {3}, "coverage", "religion"),
    case("نص عن المخرج والرزق من حيث لا يتوقع الإنسان", {3}, "coverage", "religion"),
    case("حديث يوضح أن قيمة العمل مرتبطة بالنية", {4}, "coverage", "religion"),
    case("ما النص الذي يجعل القصد أساس الجزاء؟", {4}, "coverage", "religion"),
    case("ابحث عن معنى الأعمال بالنيات", {4}, "coverage", "religion"),
    case("أفضل الناس من ينفع غيره ويخدم المجتمع", {5}, "coverage", "religion"),
    case("قول مأثور عن مساعدة الناس", {5}, "coverage", "religion"),
    case("من هو الإنسان الخيّر في النص الديني؟", {5}, "coverage", "religion"),
    case("نشاط بدني يقوي القلب ويحسن حركة الدم", {6}, "coverage", "health"),
    case("عادة رياضية تحافظ على صحة الدورة الدموية", {6}, "coverage", "health"),
    case("ما الفائدة الصحية من التمرين المنتظم؟", {6}, "coverage", "health"),
    case("نصيحة عن شرب الماء وصحة الكلى", {7}, "coverage", "health"),
    case("ما الذي يحافظ على نضارة البشرة وترطيب الجسم؟", {7}, "coverage", "health"),
    case("أهمية تناول كمية كافية من السوائل يوميا", {7}, "coverage", "health"),
    case("النوم الكافي يساعد الذاكرة والانتباه", {8}, "coverage", "health"),
    case("كم ساعة راحة يحتاجها الإنسان لتحسين التركيز؟", {8}, "coverage", "health"),
    case("علاقة جودة النوم بالأداء الذهني", {8}, "coverage", "health"),
    case("الغذاء الطازج يقلل أمراض القلب والسكري", {9}, "coverage", "health"),
    case("فوائد الخضروات والفواكه للصحة المزمنة", {9}, "coverage", "health"),
    case("أي طعام يساعد في الوقاية من السكري؟", {9}, "coverage", "health"),
    case("نماذج حديثة تتعلم الأنماط عبر شبكات عصبية عميقة", {10}, "coverage", "tech"),
    case("كيف يتعلم الذكاء الاصطناعي من البيانات؟", {10}, "coverage", "tech"),
    case("تقنية تعتمد على طبقات عصبية لفهم المعلومات", {10}, "coverage", "tech"),
    case("حواسيب جديدة قد تحل مسائل لا تستطيعها الأجهزة التقليدية", {11}, "coverage", "tech"),
    case("ما الذي تعد به الحوسبة الكمية؟", {11}, "coverage", "tech"),
    case("تقنية حاسوبية للمشكلات المعقدة جدا", {11}, "coverage", "tech"),
    case("سجل رقمي موزع يحمي المعاملات", {12}, "coverage", "tech"),
    case("تقنية تحفظ التحويلات بلا مركز واحد", {12}, "coverage", "tech"),
    case("ما معنى دفتر معاملات آمن ولا مركزي؟", {12}, "coverage", "tech"),
    case("كيف ترتب محركات البحث الصفحات بحسب الصلة؟", {13}, "coverage", "tech"),
    case("خوارزميات اختيار النتائج المناسبة للاستعلام", {13}, "coverage", "tech"),
    case("نص يشرح ترتيب نتائج البحث", {13}, "coverage", "tech"),
    case("نماذج لغوية تفهم وتكتب بلغات كثيرة", {14}, "coverage", "tech"),
    case("قدرة الأنظمة الكبيرة على توليد نصوص متعددة اللغات", {14}, "coverage", "tech"),
    case("أي تقنية تفهم العربية واللغات الأخرى بدقة؟", {14}, "coverage", "tech"),
    case("مهمة فضائية جديدة إلى المريخ تبحث عن حياة", {15}, "coverage", "science"),
    case("مسبار ناسا لاستكشاف سطح الكوكب الأحمر", {15}, "coverage", "science"),
    case("خبر علمي عن إرسال جهاز لدراسة المريخ", {15}, "coverage", "science"),
    case("كوكب بعيد خارج نظامنا الشمسي فيه ماء", {16}, "coverage", "science"),
    case("غلاف جوي غني بالماء حول جرم بعيد", {16}, "coverage", "science"),
    case("اكتشاف فلكي عن كوكب خارج المجموعة الشمسية", {16}, "coverage", "science"),
    case("لقاحات تعلم المناعة التعرف السريع على الفيروسات", {17}, "coverage", "science"),
    case("كيف تعمل لقاحات الرنا المرسال؟", {17}, "coverage", "science"),
    case("تدريب جهاز المناعة باستخدام mRNA", {17}, "coverage", "science"),
    case("بيت شعر يقول إن المطالب لا تنال بالأماني", {18}, "coverage", "poetry"),
    case("الشعر الذي يحث على أخذ الدنيا بالجهد", {18}, "coverage", "poetry"),
    case("معنى أن النجاح لا يأتي بالتمني فقط", {18}, "coverage", "poetry"),
    case("قصيدة تربط إرادة الشعوب بالحياة والقدر", {19}, "coverage", "poetry"),
    case("إذا أراد الناس الحياة فماذا يحدث؟", {19}, "coverage", "poetry"),
    case("بيت شعري عن قوة إرادة الشعب", {19}, "coverage", "poetry"),
    case("بيت المتنبي عن العزم والمكارم", {20}, "coverage", "poetry"),
    case("الشعر الذي يجعل العزائم على قدر أصحابها", {20}, "coverage", "poetry"),
    case("مقولة شعرية عن كبار النفوس والإنجازات", {20}, "coverage", "poetry"),
    case("من يفعل المعروف في غير أهله يندم", {21}, "coverage", "poetry"),
    case("بيت شعر يحذر من وضع الإحسان في غير موضعه", {21}, "coverage", "poetry"),
    case("حكمة شعرية عن سوء تقدير المعروف", {21}, "coverage", "poetry"),
    case("ارتفاع النفط بعد قرار أوبك بلس بتقليل الإنتاج", {22}, "coverage", "economy"),
    case("ما سبب صعود أسعار النفط العالمية؟", {22}, "coverage", "economy"),
    case("تأثير تخفيض إنتاج النفط على السعر", {22}, "coverage", "economy"),
    case("شركة تقنية تستغني عن موظفين ضمن إعادة هيكلة", {23}, "coverage", "economy"),
    case("خبر اقتصادي عن تسريح عاملين في قطاع التكنولوجيا", {23}, "coverage", "economy"),
    case("لماذا أعلنت الشركة الكبرى فصل آلاف الموظفين؟", {23}, "coverage", "economy"),
    case("الأسواق ترتفع بعد خفض أسعار الفائدة", {24}, "coverage", "economy"),
    case("أثر قرار البنك المركزي على البورصة", {24}, "coverage", "economy"),
    case("متى سجلت الأسواق المالية ارتفاعا قياسيا؟", {24}, "coverage", "economy"),
    case("منتخب يسجل ثلاثة أهداف ويفوز في كأس العالم", {25}, "coverage", "sports"),
    case("انتصار تاريخي في مباراة عالمية لكرة القدم", {25}, "coverage", "sports"),
    case("خبر رياضي عن فوز المنتخب بثلاثية", {25}, "coverage", "sports"),
    case("عداء يحطم الرقم العالمي في سباق قصير", {26}, "coverage", "sports"),
    case("إنجاز في سباق المئة متر خلال بطولة دولية", {26}, "coverage", "sports"),
    case("من كسر الرقم القياسي العالمي في الجري؟", {26}, "coverage", "sports"),
    case("عادة القراءة توسع المعرفة والآفاق الفكرية", {27}, "coverage", "education"),
    case("كيف تنمي القراءة عقل الإنسان؟", {27}, "coverage", "education"),
    case("أهمية المطالعة في زيادة الثقافة", {27}, "coverage", "education"),
    case("الدراسة عبر الإنترنت أصبحت أكثر تفاعلا بعد الجائحة", {28}, "coverage", "education"),
    case("تطور التعليم الإلكتروني بعد كوفيد", {28}, "coverage", "education"),
    case("كيف تغيرت أساليب التعلم عن بعد؟", {28}, "coverage", "education"),
    case("طبق عربي تقليدي مشهور بالتوابل الغنية", {29}, "coverage", "culture"),
    case("ما الطعام العربي المعروف باسم الكبسة؟", {29}, "coverage", "culture"),
    case("أشهر وجبة عربية ذات نكهة قوية", {29}, "coverage", "culture"),
    case("اليمن معروف بطرق قديمة في تقديم القهوة", {30}, "coverage", "culture"),
    case("مشروب عربي له تقاليد يمنية عمرها مئات السنين", {30}, "coverage", "culture"),
    case("ما علاقة اليمن بتاريخ القهوة العربية؟", {30}, "coverage", "culture"),
]

TEST_SET = PARAPHRASE + OBLIQUE + ADVERSARIAL + COVERAGE


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[int] | tuple[int, ...], relevant: frozenset[int], k: int) -> float:
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def reciprocal_rank(retrieved: list[int] | tuple[int, ...], relevant: frozenset[int]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[int] | tuple[int, ...], relevant: frozenset[int], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, d in enumerate(retrieved[:k], 1)
        if d in relevant
    )
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def summarize_results(results: list[EvalResult]) -> MetricRow:
    n = len(results)
    return {
        "MRR": sum(r.reciprocal_rank for r in results) / n,
        "NDCG@5": sum(r.ndcg_at_5 for r in results) / n,
        "P@1": sum(r.p_at_1 for r in results) / n,
        "P@3": sum(r.p_at_3 for r in results) / n,
    }


# ---------------------------------------------------------------------------
# Per-dimension evaluation
# ---------------------------------------------------------------------------

def encode_test_queries(
    searcher: ArabicSearcher,
    cases: list[EvalCase] = TEST_SET,
) -> list[EncodedQuery]:
    vectors = searcher.encode_queries([row.query for row in cases])
    return [(vectors[i], row) for i, row in enumerate(cases)]


def evaluate_cases(
    searcher: ArabicSearcher,
    dim: int,
    encoded_queries: list[EncodedQuery],
) -> list[EvalResult]:
    results = []
    for query_vector, row in encoded_queries:
        hits = searcher.search_by_vector(query_vector, top_k=K, dim=dim)
        retrieved = tuple(r["id"] for r in hits)
        results.append(
            EvalResult(
                case=row,
                retrieved=retrieved,
                reciprocal_rank=reciprocal_rank(retrieved, row.relevant),
                ndcg_at_5=ndcg_at_k(retrieved, row.relevant, K),
                p_at_1=precision_at_k(retrieved, row.relevant, 1),
                p_at_3=precision_at_k(retrieved, row.relevant, 3),
            )
        )
    return results


def evaluate_dim(
    searcher: ArabicSearcher,
    dim: int,
    encoded_queries: list[EncodedQuery] | None = None,
) -> MetricRow:
    query_rows = encode_test_queries(searcher) if encoded_queries is None else encoded_queries
    return summarize_results(evaluate_cases(searcher, dim, query_rows))


def grouped_metrics(
    results: list[EvalResult],
    key: Callable[[EvalResult], str],
) -> dict[str, MetricRow]:
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for row in results:
        groups[key(row)].append(row)
    return {
        group: {**summarize_results(rows), "n": float(len(rows))}
        for group, rows in groups.items()
    }


def print_metric_table(rows: list[tuple[int, MetricRow, float]]) -> None:
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


def print_group_table(title: str, rows: dict[str, MetricRow]) -> None:
    width = 76
    print(f"\n{title}")
    print("-" * width)
    print(f"  {'Group':<14} {'n':>4}    {'MRR':>6}   {'NDCG@5':>7}    {'P@1':>5}    {'P@3':>5}")
    for name, metrics in sorted(rows.items()):
        print(
            f"  {name:<14} {int(metrics['n']):>4}    {metrics['MRR']:>6.3f}   "
            f"{metrics['NDCG@5']:>7.3f}    {metrics['P@1']:>5.3f}    {metrics['P@3']:>5.3f}"
        )
    print("-" * width)


def print_error_analysis(results: list[EvalResult], limit: int = 12) -> None:
    top1_misses = [
        row for row in results
        if row.retrieved[0] not in row.case.relevant
    ]
    no_top_k_hit = [
        row for row in results
        if not any(doc_id in row.case.relevant for doc_id in row.retrieved[:K])
    ]
    print(
        f"\nError analysis @ {max(SUPPORTED_DIMS)} dims: "
        f"top-1 misses={len(top1_misses)}/{len(results)}, "
        f"no top-{K} hit={len(no_top_k_hit)}/{len(results)}"
    )
    if not top1_misses:
        print("No top-1 misses.")
        return

    print(f"Showing first {min(limit, len(top1_misses))} top-1 misses:")
    for row in top1_misses[:limit]:
        expected = ",".join(str(doc_id) for doc_id in sorted(row.case.relevant))
        got = ",".join(str(doc_id) for doc_id in row.retrieved[:K]) or "-"
        print(
            f"  - [{row.case.category}/{row.case.tier}] "
            f"expected {expected}; got {got}; {row.case.query}"
        )


def test_set_summary() -> str:
    tier_counts = Counter(row.tier for row in TEST_SET)
    parts = " + ".join(f"{count} {tier}" for tier, count in sorted(tier_counts.items()))
    return f"n={len(TEST_SET)} queries ({parts}) · k={K} · binary relevance"


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

    print("Encoding evaluation queries once...")
    t0 = time.perf_counter()
    encoded_queries = encode_test_queries(searcher)
    query_encode_time = time.perf_counter() - t0
    print(f"Encoded {len(encoded_queries)} queries in {query_encode_time:.1f}s\n")

    print("Evaluating across Matryoshka dimensions...\n")
    rows = []
    detailed_results: dict[int, list[EvalResult]] = {}
    for dim in sorted(SUPPORTED_DIMS, reverse=True):
        t = time.perf_counter()
        results = evaluate_cases(searcher, dim, encoded_queries)
        detailed_results[dim] = results
        rows.append((dim, summarize_results(results), time.perf_counter() - t))

    print_metric_table(rows)
    default_dim = max(SUPPORTED_DIMS)
    default_results = detailed_results[default_dim]
    print_group_table(
        f"Tier breakdown @ {default_dim} dims",
        grouped_metrics(default_results, lambda row: row.case.tier),
    )
    print_group_table(
        f"Category breakdown @ {default_dim} dims",
        grouped_metrics(default_results, lambda row: row.case.category),
    )
    print_error_analysis(default_results)
    print(f"\n{test_set_summary()}")


if __name__ == "__main__":
    main()
