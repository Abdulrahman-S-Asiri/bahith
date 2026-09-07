"""Reproducible retrieval evaluation for Bahith.

The embedded 120-query suite is retained as regression-only evidence. External
JSONL cases can be supplied for independently reviewed evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    query: str
    relevant: frozenset[int | str]
    tier: str
    category: str
    case_id: str = ""
    split: str = "regression"
    document_id: int | str | None = None


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    retrieved: tuple[int | str, ...]
    reciprocal_rank: float
    ndcg_at_5: float
    p_at_1: float
    p_at_3: float
    recall_at_5: float
    success_at_3: float
    success_at_5: float


EncodedQuery = tuple[Any, EvalCase]
MetricRow = dict[str, float | int | None]
K = 5  # retrieval depth used for all metrics
VALID_SPLITS = frozenset({"development", "calibration", "test", "regression"})
VALID_METHODS = frozenset({"semantic", "keyword", "hybrid", "cascade"})


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

LEGACY_REGRESSION_SET = PARAPHRASE + OBLIQUE + ADVERSARIAL + COVERAGE
# Compatibility alias. This set is regression-only, not independent evidence.
TEST_SET = LEGACY_REGRESSION_SET


# ---------------------------------------------------------------------------
# Metrics and aggregation
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved: list[int | str] | tuple[int | str, ...],
    relevant: frozenset[int | str],
    k: int,
) -> float:
    return sum(1 for doc_id in retrieved[:k] if doc_id in relevant) / k


def reciprocal_rank(
    retrieved: list[int | str] | tuple[int | str, ...],
    relevant: frozenset[int | str],
    k: int = K,
) -> float:
    for rank, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    retrieved: list[int | str] | tuple[int | str, ...],
    relevant: frozenset[int | str],
    k: int,
) -> float:
    if not relevant:
        raise ValueError("recall is undefined for a no-answer query")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def success_at_k(
    retrieved: list[int | str] | tuple[int | str, ...],
    relevant: frozenset[int | str],
    k: int,
) -> float:
    if not relevant:
        raise ValueError("success is undefined for a no-answer query")
    return float(any(doc_id in relevant for doc_id in retrieved[:k]))


def ndcg_at_k(
    retrieved: list[int | str] | tuple[int | str, ...],
    relevant: frozenset[int | str],
    k: int,
) -> float:
    if not relevant:
        raise ValueError("NDCG is undefined for a no-answer query")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], 1)
        if doc_id in relevant
    )
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg


def _make_result(row: EvalCase, hits: list[dict[str, Any]]) -> EvalResult:
    try:
        retrieved = tuple(hit["id"] for hit in hits[:K])
    except (KeyError, TypeError) as exc:
        raise ValueError("every retrieval hit must be an object with an id") from exc
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("retrieval results contain duplicate document ids")
    if not row.relevant:
        return EvalResult(row, retrieved, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return EvalResult(
        case=row,
        retrieved=retrieved,
        reciprocal_rank=reciprocal_rank(retrieved, row.relevant, K),
        ndcg_at_5=ndcg_at_k(retrieved, row.relevant, K),
        p_at_1=precision_at_k(retrieved, row.relevant, 1),
        p_at_3=precision_at_k(retrieved, row.relevant, 3),
        recall_at_5=recall_at_k(retrieved, row.relevant, K),
        success_at_3=success_at_k(retrieved, row.relevant, 3),
        success_at_5=success_at_k(retrieved, row.relevant, K),
    )


def summarize_results(results: list[EvalResult]) -> MetricRow:
    answerable = [row for row in results if row.case.relevant]
    no_answer_count = len(results) - len(answerable)
    if not answerable:
        return {
            "n": 0,
            "excluded_no_answer": no_answer_count,
            "MRR@5": None,
            "NDCG@5": None,
            "P@1": None,
            "P@3": None,
            "Recall@5": None,
            "Success@3": None,
            "Success@5": None,
        }
    n = len(answerable)
    return {
        "n": n,
        "excluded_no_answer": no_answer_count,
        "MRR@5": sum(row.reciprocal_rank for row in answerable) / n,
        "NDCG@5": sum(row.ndcg_at_5 for row in answerable) / n,
        "P@1": sum(row.p_at_1 for row in answerable) / n,
        "P@3": sum(row.p_at_3 for row in answerable) / n,
        "Recall@5": sum(row.recall_at_5 for row in answerable) / n,
        "Success@3": sum(row.success_at_3 for row in answerable) / n,
        "Success@5": sum(row.success_at_5 for row in answerable) / n,
    }


def summarize_no_answer(results: list[EvalResult]) -> dict[str, float | int | None]:
    rows = [row for row in results if not row.case.relevant]
    if not rows:
        return {"n": 0, "abstention_rate": None, "returned_any_rate": None}
    abstained = sum(not row.retrieved for row in rows)
    return {
        "n": len(rows),
        "abstention_rate": abstained / len(rows),
        "returned_any_rate": 1.0 - (abstained / len(rows)),
    }


def grouped_no_answer(
    results: list[EvalResult],
    key: Callable[[EvalResult], str],
) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for row in results:
        if not row.case.relevant:
            groups[key(row)].append(row)
    return {group: summarize_no_answer(rows) for group, rows in sorted(groups.items())}


def grouped_metrics(
    results: list[EvalResult],
    key: Callable[[EvalResult], str],
) -> dict[str, MetricRow]:
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for row in results:
        groups[key(row)].append(row)
    return {group: summarize_results(rows) for group, rows in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_cases_sha256(cases: list[EvalCase]) -> str:
    rows = [
        {
            "id": row.case_id,
            "query": row.query,
            "relevant": sorted(row.relevant, key=str),
            "tier": row.tier,
            "category": row.category,
            "split": row.split,
            "document_id": row.document_id,
        }
        for row in cases
    ]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_corpus(corpus: Any) -> set[int | str]:
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("corpus must be a non-empty JSON array")
    ids: list[int | str] = []
    for index, doc in enumerate(corpus, 1):
        if not isinstance(doc, dict):
            raise ValueError(f"corpus item {index} must be an object")
        doc_id = doc.get("id")
        if isinstance(doc_id, bool) or not isinstance(doc_id, int):
            raise ValueError(f"corpus item {index} id must be an integer")
        if not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise ValueError(f"corpus item {index} has empty text")
        document_id = doc.get("document_id")
        if document_id is not None and (
            isinstance(document_id, bool)
            or not isinstance(document_id, (int, str))
            or document_id == ""
        ):
            raise ValueError(f"corpus item {index} has an invalid document_id")
        ids.append(doc_id)
    if len(ids) != len(set(ids)):
        duplicates = [str(doc_id) for doc_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"duplicate corpus ids: {', '.join(duplicates)}")
    return set(ids)


def _normalized_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def validate_cases(
    cases: list[EvalCase],
    corpus_ids: set[int | str],
    corpus_document_ids: set[int | str] | None = None,
) -> None:
    if not cases:
        raise ValueError("evaluation cases are empty")
    case_ids: set[str] = set()
    queries: dict[str, str] = {}
    for index, row in enumerate(cases, 1):
        if not row.case_id:
            raise ValueError(f"case {index} is missing id")
        if row.case_id in case_ids:
            raise ValueError(f"duplicate case id: {row.case_id}")
        case_ids.add(row.case_id)
        normalized = _normalized_query(row.query)
        if not normalized:
            raise ValueError(f"case {row.case_id} has an empty query")
        if normalized in queries:
            raise ValueError(
                f"duplicate normalized query in {queries[normalized]} and {row.case_id}"
            )
        queries[normalized] = row.case_id
        if not row.tier or not row.category:
            raise ValueError(f"case {row.case_id} requires tier and category")
        if row.split not in VALID_SPLITS:
            raise ValueError(
                f"case {row.case_id} split must be one of {sorted(VALID_SPLITS)}"
            )
        missing = row.relevant - corpus_ids
        if missing:
            raise ValueError(
                f"case {row.case_id} references absent corpus ids: "
                + ", ".join(map(str, sorted(missing, key=str)))
            )
        if row.document_id is not None:
            if isinstance(row.document_id, bool) or not isinstance(row.document_id, (int, str)):
                raise ValueError(f"case {row.case_id} has an invalid document_id")
            if row.document_id not in (corpus_document_ids or set()):
                raise ValueError(
                    f"case {row.case_id} document_id {row.document_id!r} is absent from "
                    "corpus document_id fields"
                )


def _review_status(metadata: dict[str, Any]) -> str:
    review = metadata.get("human_review")
    if not isinstance(review, dict):
        return "UNREVIEWED"
    status = review.get("status", "UNREVIEWED")
    if status == "UNREVIEWED":
        return status
    if status != "HUMAN_REVIEWED":
        raise ValueError("human_review.status must be UNREVIEWED or HUMAN_REVIEWED")
    required = ("reviewer", "reviewed_at", "protocol")
    if not all(isinstance(review.get(field), str) and review[field].strip() for field in required):
        raise ValueError(
            "HUMAN_REVIEWED metadata requires reviewer, reviewed_at, and protocol"
        )
    return status


def load_cases_jsonl(path: str | Path) -> tuple[list[EvalCase], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on cases line {line_number}: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"cases line {line_number} must be an object")
            if "_meta" in item:
                if cases or metadata or set(item) != {"_meta"} or not isinstance(item["_meta"], dict):
                    raise ValueError("_meta must be the first and only field on its line")
                metadata = item["_meta"]
                continue
            required = ("id", "query", "tier", "category", "split")
            missing_fields = [field for field in required if field not in item]
            if missing_fields:
                raise ValueError(
                    f"cases line {line_number} missing fields: {', '.join(missing_fields)}"
                )
            relevant_raw = item.get("relevant_ids", [])
            if not isinstance(relevant_raw, list) or any(
                isinstance(doc_id, bool) or not isinstance(doc_id, (int, str))
                for doc_id in relevant_raw
            ):
                raise ValueError(f"cases line {line_number} relevant_ids must be an id array")
            if len(relevant_raw) != len(set(relevant_raw)):
                raise ValueError(f"cases line {line_number} has duplicate relevant_ids")
            for field in required:
                if not isinstance(item[field], str) or not item[field].strip():
                    raise ValueError(f"cases line {line_number} field {field} must be nonempty text")
            no_answer = item.get("no_answer", False)
            if not isinstance(no_answer, bool):
                raise ValueError(f"cases line {line_number} no_answer must be boolean")
            if bool(relevant_raw) == no_answer:
                raise ValueError(
                    f"cases line {line_number} must have relevant_ids or no_answer=true, not both"
                )
            cases.append(
                EvalCase(
                    query=str(item["query"]),
                    relevant=frozenset(relevant_raw),
                    tier=str(item["tier"]),
                    category=str(item["category"]),
                    case_id=str(item["id"]),
                    split=str(item["split"]),
                    document_id=item.get("document_id"),
                )
            )
    metadata = dict(metadata)
    metadata["human_review_status"] = _review_status(metadata)
    return cases, metadata


def load_corpus_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        corpus = json.load(handle)
    validate_corpus(corpus)
    return corpus


def legacy_cases() -> list[EvalCase]:
    return [
        replace(row, case_id=f"legacy-{index:03d}")
        for index, row in enumerate(LEGACY_REGRESSION_SET, 1)
    ]


# ---------------------------------------------------------------------------
# Retrieval and statistical comparison
# ---------------------------------------------------------------------------

def encode_test_queries(searcher: Any, cases: list[EvalCase] | None = None) -> list[EncodedQuery]:
    rows = legacy_cases() if cases is None else cases
    vectors = searcher.encode_queries([row.query for row in rows])
    return [(vectors[index], row) for index, row in enumerate(rows)]


def evaluate_cases(
    searcher: Any,
    dim: int,
    encoded_queries: list[EncodedQuery],
) -> list[EvalResult]:
    return [_make_result(row, hits) for row, hits in _retrieve_encoded(searcher, dim, encoded_queries)]


def _retrieve_encoded(
    searcher: Any,
    dim: int,
    encoded_queries: list[EncodedQuery],
) -> list[tuple[EvalCase, list[dict[str, Any]]]]:
    rows = []
    for query_vector, row in encoded_queries:
        if row.document_id is None:
            hits = searcher.search_by_vector(query_vector, top_k=K, dim=dim)
        else:
            hits = searcher.search_by_vector(
                query_vector,
                top_k=K,
                dim=dim,
                document_id=row.document_id,
            )
        rows.append((row, hits))
    return rows


def evaluate_retrieval_cases(
    searcher: Any,
    dim: int,
    cases: list[EvalCase],
    method: str,
    candidate_k: int = 50,
) -> list[EvalResult]:
    return [
        _make_result(row, hits)
        for row, hits in _retrieve_text(searcher, dim, cases, method, candidate_k)
    ]


def _retrieve_text(
    searcher: Any,
    dim: int,
    cases: list[EvalCase],
    method: str,
    candidate_k: int,
) -> list[tuple[EvalCase, list[dict[str, Any]]]]:
    rows = []
    for row in cases:
        hits = searcher.retrieve(
            row.query,
            top_k=K,
            dim=dim,
            method=method,
            candidate_k=candidate_k,
            document_id=row.document_id,
        )
        rows.append((row, hits))
    return rows


def evaluate_dim(
    searcher: Any,
    dim: int,
    encoded_queries: list[EncodedQuery] | None = None,
) -> MetricRow:
    query_rows = encode_test_queries(searcher) if encoded_queries is None else encoded_queries
    return summarize_results(evaluate_cases(searcher, dim, query_rows))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(samples: list[float], queries_per_batch: int) -> dict[str, float | int | str]:
    if not samples:
        raise ValueError("timing samples are empty")
    median = statistics.median(samples)
    return {
        "repeats": len(samples),
        "scope": "complete query-set batch",
        "queries_per_batch": queries_per_batch,
        "median_batch_seconds": median,
        "p95_batch_seconds": _percentile(samples, 0.95),
        "median_queries_per_second": queries_per_batch / median if median > 0 else 0.0,
    }


QUALITY_FIELDS: dict[str, Callable[[EvalResult], float]] = {
    "MRR@5": lambda row: row.reciprocal_rank,
    "NDCG@5": lambda row: row.ndcg_at_5,
    "Recall@5": lambda row: row.recall_at_5,
    "Success@3": lambda row: row.success_at_3,
    "Success@5": lambda row: row.success_at_5,
}


def paired_bootstrap_difference(
    candidate: list[EvalResult],
    reference: list[EvalResult],
    metric: Callable[[EvalResult], float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    candidate_rows = [row for row in candidate if row.case.relevant]
    reference_rows = [row for row in reference if row.case.relevant]
    candidate_ids = [row.case.case_id for row in candidate_rows]
    reference_ids = [row.case.case_id for row in reference_rows]
    if candidate_ids != reference_ids:
        raise ValueError("paired bootstrap inputs must contain aligned answerable cases")
    if not candidate_rows:
        raise ValueError("paired bootstrap needs at least one answerable case")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    differences = [
        metric(candidate_row) - metric(reference_row)
        for candidate_row, reference_row in zip(candidate_rows, reference_rows)
    ]
    observed = statistics.fmean(differences)
    rng = random.Random(seed)
    n = len(differences)
    samples = [
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    ]
    return {
        "n": n,
        "resamples": resamples,
        "candidate_minus_1024": observed,
        "ci95_low": _percentile(samples, 0.025),
        "ci95_high": _percentile(samples, 0.975),
    }


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_comparisons(
    candidate: list[EvalResult],
    reference: list[EvalResult],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    def compare_group(
        candidate_group: list[EvalResult],
        reference_group: list[EvalResult],
        label: str,
    ) -> dict[str, Any]:
        if not any(row.case.relevant for row in candidate_group):
            return {"status": "no_answerable_cases"}
        return {
            metric_name: paired_bootstrap_difference(
                candidate_group,
                reference_group,
                metric,
                resamples=resamples,
                seed=_stable_seed(seed, label, metric_name),
            )
            for metric_name, metric in QUALITY_FIELDS.items()
        }

    output: dict[str, Any] = {
        "overall": compare_group(candidate, reference, "overall"),
        "caveat": (
            "Queries are resampled as independent units. Shared relevant documents can "
            "correlate errors, so these intervals may be too narrow."
        ),
    }
    for grouping, getter in (
        ("tier", lambda row: row.case.tier),
        ("category", lambda row: row.case.category),
    ):
        labels = sorted({getter(row) for row in candidate})
        output[f"by_{grouping}"] = {}
        for label in labels:
            candidate_group = [row for row in candidate if getter(row) == label]
            reference_group = [row for row in reference if getter(row) == label]
            output[f"by_{grouping}"][label] = compare_group(
                candidate_group,
                reference_group,
                f"{grouping}:{label}",
            )
    return output


def _result_record(row: EvalResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": row.case.case_id,
        "query": row.case.query,
        "split": row.case.split,
        "tier": row.case.tier,
        "category": row.case.category,
        "relevant_ids": sorted(row.case.relevant, key=str),
        "no_answer": not bool(row.case.relevant),
        "retrieved_ids": list(row.retrieved),
    }
    if row.case.relevant:
        record["metrics"] = {
            "MRR@5": row.reciprocal_rank,
            "NDCG@5": row.ndcg_at_5,
            "P@1": row.p_at_1,
            "P@3": row.p_at_3,
            "Recall@5": row.recall_at_5,
            "Success@3": row.success_at_3,
            "Success@5": row.success_at_5,
        }
    else:
        record["no_answer_outcome"] = "abstained" if not row.retrieved else "returned_results"
    return record


def _configuration_result(
    results: list[EvalResult],
    warm_timings: list[float],
    cold_seconds: float,
    warmup_runs: int,
    rankings_consistent: bool,
) -> dict[str, Any]:
    return {
        "answerable": summarize_results(results),
        "no_answer": summarize_no_answer(results),
        "no_answer_by_tier": grouped_no_answer(results, lambda row: row.case.tier),
        "no_answer_by_category": grouped_no_answer(results, lambda row: row.case.category),
        "by_tier": grouped_metrics(results, lambda row: row.case.tier),
        "by_category": grouped_metrics(results, lambda row: row.case.category),
        "timing": {
            "cache_cold_batch_seconds": cold_seconds,
            "warmup_runs_including_cold": warmup_runs,
            "warm_repetitions": timing_summary(warm_timings, len(results)),
        },
        "rankings_consistent_across_repeats": rankings_consistent,
        "cases": [_result_record(row) for row in results],
    }


def run_evaluation(
    searcher: Any,
    cases: list[EvalCase],
    *,
    dimensions: list[int],
    methods: list[str],
    repeats: int,
    seed: int,
    bootstrap_resamples: int,
    candidate_k: int = 50,
    warmup_runs: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    if candidate_k < K:
        raise ValueError(f"candidate_k must be at least {K}")
    if warmup_runs < 1:
        raise ValueError("warmup_runs must be at least one")
    if not dimensions or len(dimensions) != len(set(dimensions)):
        raise ValueError("dimensions must be a non-empty list without duplicates")
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a non-empty list without duplicates")
    invalid_methods = sorted(set(methods) - VALID_METHODS)
    if invalid_methods:
        raise ValueError(f"unsupported methods: {', '.join(invalid_methods)}")

    encoded: list[EncodedQuery] | None = None
    encoding_seconds: float | None = None
    if "semantic" in methods:
        started = time.perf_counter()
        encoded = encode_test_queries(searcher, cases)
        encoding_seconds = time.perf_counter() - started

    results_by_method: dict[str, dict[str, Any]] = {}
    raw: dict[tuple[str, int], list[EvalResult]] = {}
    for method in methods:
        results_by_method[method] = {}
        for dim in dimensions:
            _reset_query_cache(searcher)
            cold_started = time.perf_counter()
            if method == "semantic":
                assert encoded is not None
                cold_rows = _retrieve_encoded(searcher, dim, encoded)
            else:
                cold_rows = _retrieve_text(searcher, dim, cases, method, candidate_k)
            cold_seconds = time.perf_counter() - cold_started
            cold_results = [_make_result(row, hits) for row, hits in cold_rows]

            for _ in range(warmup_runs - 1):
                if method == "semantic":
                    assert encoded is not None
                    _retrieve_encoded(searcher, dim, encoded)
                else:
                    _retrieve_text(searcher, dim, cases, method, candidate_k)

            warm_timings: list[float] = []
            quality_results: list[EvalResult] | None = None
            rankings_consistent = True
            for _ in range(repeats):
                started = time.perf_counter()
                if method == "semantic":
                    assert encoded is not None
                    retrieved_rows = _retrieve_encoded(searcher, dim, encoded)
                else:
                    retrieved_rows = _retrieve_text(
                        searcher,
                        dim,
                        cases,
                        method,
                        candidate_k,
                    )
                warm_timings.append(time.perf_counter() - started)
                current = [_make_result(row, hits) for row, hits in retrieved_rows]
                if quality_results is None:
                    quality_results = current
                elif [row.retrieved for row in current] != [
                    row.retrieved for row in quality_results
                ]:
                    rankings_consistent = False
            if quality_results is not None and [row.retrieved for row in cold_results] != [
                row.retrieved for row in quality_results
            ]:
                rankings_consistent = False
            assert quality_results is not None
            raw[(method, dim)] = quality_results
            results_by_method[method][str(dim)] = _configuration_result(
                quality_results,
                warm_timings,
                cold_seconds,
                warmup_runs,
                rankings_consistent,
            )

    comparisons: dict[str, Any] = {}
    for method in methods:
        if (method, 1024) not in raw:
            comparisons[method] = {"status": "reference_dimension_1024_not_run"}
            continue
        comparisons[method] = {}
        for dim in dimensions:
            if dim == 1024:
                continue
            comparisons[method][str(dim)] = bootstrap_comparisons(
                raw[(method, dim)],
                raw[(method, 1024)],
                resamples=bootstrap_resamples,
                seed=_stable_seed(seed, method, str(dim)),
            )

    timing_meta = {
        "query_encoding_seconds": encoding_seconds,
        "quality_metrics_exclude_all_timings": True,
        "semantic_ranking_excludes_query_encoding": True,
        "other_method_timing_scope": (
            "keyword measures retrieval only; hybrid/cascade measure their complete retrieve() "
            "pipeline because the public API accepts query text"
        ),
        "cache_policy": {
            "query_cache_size": getattr(searcher, "query_cache_size", None),
            "reset_before_each_method_dimension": True,
            "cold_pass_reported_separately": True,
            "warmup_runs_including_cold": warmup_runs,
            "warm_repetitions_exclude_warmup": True,
        },
    }
    return {
        "results": results_by_method,
        "paired_bootstrap_vs_1024": comparisons,
    }, timing_meta


# ---------------------------------------------------------------------------
# Evidence metadata and CLI
# ---------------------------------------------------------------------------

def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "sentence-transformers", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _set_runtime_seed(seed: int, *, include_torch: bool) -> None:
    random.seed(seed)
    numpy_module = sys.modules.get("numpy")
    if numpy_module is not None:
        numpy_module.random.seed(seed % (2**32))
    if include_torch:
        torch_module = sys.modules.get("torch")
        if torch_module is not None:
            torch_module.manual_seed(seed)
            try:
                if torch_module.cuda.is_available():
                    torch_module.cuda.manual_seed_all(seed)
            except (AttributeError, RuntimeError):
                pass


def _reset_query_cache(searcher: Any) -> None:
    public_reset = getattr(searcher, "clear_query_cache", None)
    if callable(public_reset):
        public_reset()
        return
    query_cache = vars(searcher).get("_query_cache")
    if query_cache is None:
        return
    lock = vars(searcher).get("_lock")
    if lock is None:
        query_cache.clear()
        return
    with lock:
        query_cache.clear()


def _hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
    }
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            cuda_available = bool(torch_module.cuda.is_available())
            metadata["cuda_available"] = cuda_available
            metadata["cuda_device"] = (
                torch_module.cuda.get_device_name(0) if cuda_available else None
            )
        except (AttributeError, RuntimeError):
            metadata["cuda_available"] = None
            metadata["cuda_device"] = None
    return metadata


def _model_metadata(searcher: Any) -> dict[str, Any]:
    return {
        "name": getattr(searcher, "model_name", None),
        "revision": getattr(searcher, "model_revision", None),
        "query_prompt": getattr(searcher, "query_prompt", None),
    }


def _embedding_memory(searcher: Any) -> dict[str, Any]:
    # Read storage fields directly: the public embeddings property may lazily load
    # the dense model, which a keyword-only evaluation must never do.
    full = vars(searcher).get("_embeddings")
    by_dim = vars(searcher).get("_embeddings_by_dim", {})
    full_bytes = int(getattr(full, "nbytes", 0))
    unique_cache_arrays: list[Any] = []
    numpy_module = sys.modules.get("numpy")
    for array in by_dim.values():
        existing = ([full] if full is not None else []) + unique_cache_arrays
        overlaps = any(array is seen for seen in existing)
        if not overlaps and numpy_module is not None:
            try:
                overlaps = any(numpy_module.shares_memory(array, seen) for seen in existing)
            except (TypeError, ValueError):
                overlaps = False
        if not overlaps:
            unique_cache_arrays.append(array)
    cache_bytes = sum(int(getattr(array, "nbytes", 0)) for array in unique_cache_arrays)
    return {
        "full_embedding_matrix_bytes": full_bytes,
        "additional_unique_per_dimension_cache_bytes": cache_bytes,
        "known_embedding_arrays_total_bytes": full_bytes + cache_bytes,
        "scope_note": (
            "Array payload bytes only. This excludes Python/runtime overhead and query caches; "
            "a shared full-vector process does not shrink process RAM when evaluating a lower dim."
        ),
    }


def _parse_csv(values: list[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None, help="UTF-8 corpus JSON file")
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="UTF-8 evaluation JSONL; omit to run the embedded regression-only cases",
    )
    parser.add_argument("--output", type=str, required=True, help="output JSON path, or - for stdout")
    parser.add_argument("--dimensions", nargs="+", default=["64,128,256,512,768,1024"])
    parser.add_argument("--methods", nargs="+", default=["semantic"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="warmup passes per method/dimension; the first is reported separately as cold",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="optional development, calibration, test, or regression split filter",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    from search import ArabicSearcher, DEFAULT_CORPUS, SUPPORTED_DIMS

    corpus_path = Path(args.corpus or DEFAULT_CORPUS).resolve()
    corpus_started = time.perf_counter()
    corpus = load_corpus_json(corpus_path)
    corpus_load_seconds = time.perf_counter() - corpus_started
    corpus_ids = validate_corpus(corpus)

    if args.cases is None:
        cases = legacy_cases()
        cases_path: str | None = None
        cases_sha = _canonical_cases_sha256(cases)
        case_metadata: dict[str, Any] = {
            "name": "embedded-legacy-regression",
            "evidence_scope": "REGRESSION_ONLY",
            "human_review_status": "UNREVIEWED",
            "warning": (
                "Hand-authored against the demo corpus; not independent production evidence."
            ),
        }
    else:
        resolved_cases = Path(args.cases).resolve()
        cases, case_metadata = load_cases_jsonl(resolved_cases)
        cases_path = str(resolved_cases)
        cases_sha = sha256_file(resolved_cases)

    split_filter = set(_parse_csv(args.splits)) if args.splits else None
    if split_filter:
        invalid_splits = sorted(split_filter - VALID_SPLITS)
        if invalid_splits:
            raise ValueError(f"unsupported splits: {', '.join(invalid_splits)}")
        cases = [row for row in cases if row.split in split_filter]
    corpus_document_ids = {
        doc["document_id"] for doc in corpus if doc.get("document_id") is not None
    }
    validate_cases(cases, corpus_ids, corpus_document_ids)

    dimensions = [int(value) for value in _parse_csv(args.dimensions)]
    unsupported_dims = sorted(set(dimensions) - set(SUPPORTED_DIMS))
    if unsupported_dims:
        raise ValueError(f"unsupported dimensions: {unsupported_dims}")
    methods = _parse_csv(args.methods)
    invalid_methods = sorted(set(methods) - VALID_METHODS)
    if invalid_methods:
        raise ValueError(f"unsupported methods: {', '.join(invalid_methods)}")

    _set_runtime_seed(args.seed, include_torch=False)
    boot_started = time.perf_counter()
    searcher_kwargs: dict[str, Any] = {}
    if set(methods) == {"keyword"}:
        searcher_kwargs["load_embeddings"] = False
    searcher = ArabicSearcher(corpus, **searcher_kwargs)
    searcher_boot_seconds = time.perf_counter() - boot_started
    _set_runtime_seed(args.seed, include_torch=True)

    evaluation, timing_meta = run_evaluation(
        searcher,
        cases,
        dimensions=dimensions,
        methods=methods,
        repeats=args.repeats,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        candidate_k=args.candidate_k,
        warmup_runs=args.warmup_runs,
    )
    evidence = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dimensions": dimensions,
            "methods": methods,
            "top_k": K,
            "candidate_k": args.candidate_k,
            "repeats": args.repeats,
            "warmup_runs": args.warmup_runs,
            "seed": args.seed,
            "seed_scope": "Python, NumPy, and loaded Torch RNGs; backend determinism is not guaranteed",
            "bootstrap_resamples": args.bootstrap_resamples,
            "splits": sorted(split_filter) if split_filter else "all",
        },
        "provenance": {
            "source_files": {
                "evaluate.py_sha256": sha256_file(Path(__file__).resolve()),
                "search.py_sha256": sha256_file(Path(__file__).resolve().with_name("search.py")),
                "note": "Source hashes identify the evaluated code even when Git HEAD is older.",
            },
            "corpus": {
                "path": str(corpus_path),
                "sha256": sha256_file(corpus_path),
                "documents": len(corpus),
            },
            "cases": {
                "path": cases_path,
                "sha256": cases_sha,
                "queries": len(cases),
                "split_counts": dict(sorted(Counter(row.split for row in cases).items())),
                "metadata": case_metadata,
            },
            "model": _model_metadata(searcher),
            "runtime": {
                "python": platform.python_version(),
                "dependencies": _dependency_versions(),
                "hardware": _hardware_metadata(),
            },
            "embedding_memory": _embedding_memory(searcher),
        },
        "timing": {
            "corpus_load_seconds": corpus_load_seconds,
            "searcher_boot_seconds": searcher_boot_seconds,
            **timing_meta,
        },
        **evaluation,
    }
    payload = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        print(payload, end="")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(
            f"Wrote {len(cases)}-query evaluation evidence to {output_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
