"""FastAPI web app for بَاحث Arabic semantic search."""
from __future__ import annotations

import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from search import (
    DEFAULT_DIM,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    SUPPORTED_DIMS,
    ArabicSearcher,
    load_corpus,
)

ROOT = Path(__file__).parent
templates = Jinja2Templates(directory=ROOT / "templates")

CATEGORY_LABELS_AR = {
    "religion":  "دين",
    "health":    "صحة",
    "tech":      "تقنية",
    "science":   "علوم",
    "poetry":    "شعر",
    "economy":   "اقتصاد",
    "sports":    "رياضة",
    "education": "تعليم",
    "culture":   "ثقافة",
}

SUGGESTIONS = [
    "ما فوائد القراءة وتطوير العقل؟",
    "أهمية الذكاء الاصطناعي في حياتنا",
    "أبيات في العزيمة والإصرار",
    "نصائح للحفاظ على الصحة",
    "آخر اكتشافات الفضاء",
    "أطباق عربية تقليدية",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model load is the expensive part (~1 minute cold), so we do it once
    # at startup and reuse the instance for every request.
    print("Loading model...")
    t0 = time.perf_counter()
    corpus = load_corpus()
    app.state.searcher = ArabicSearcher(corpus)
    app.state.corpus = corpus
    print(f"Ready in {time.perf_counter() - t0:.1f}s · {len(corpus)} docs")
    yield


app = FastAPI(title="Bāḥith", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def label_ar(category: str) -> str:
    return CATEGORY_LABELS_AR.get(category, category)


def run_search(request: Request, q: str, dim: int, top_k: int) -> dict:
    # User input arrives via query string, so guard against bad values.
    if dim not in SUPPORTED_DIMS:
        dim = DEFAULT_DIM
    top_k = max(1, min(MAX_TOP_K, top_k))

    if not q.strip():
        return {
            "q": q,
            "results": [],
            "embedding": None,
            "elapsed_ms": 0.0,
            "dim": dim,
            "top_k": top_k,
        }

    t0 = time.perf_counter()
    query_vector = request.app.state.searcher.encode_query(q)
    results = request.app.state.searcher.search_by_vector(query_vector, top_k=top_k, dim=dim)
    embedding = request.app.state.searcher.explain_vector(query_vector, dim=dim)
    return {
        "q": q,
        "results": results,
        "embedding": embedding,
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
        "dim": dim,
        "top_k": top_k,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request, "home.html", {"suggestions": SUGGESTIONS},
    )


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = "",
    dim: int = DEFAULT_DIM,
    top_k: int = DEFAULT_TOP_K,
):
    payload = run_search(request, q, dim, top_k)
    return templates.TemplateResponse(
        request,
        "search.html",
        {
            **payload,
            "supported_dims": SUPPORTED_DIMS,
            "corpus_size": len(request.app.state.corpus),
        },
    )


@app.get("/api/search", response_class=HTMLResponse)
async def api_search(
    request: Request,
    q: str = "",
    dim: int = DEFAULT_DIM,
    top_k: int = DEFAULT_TOP_K,
):
    # HTMX endpoint: returns just the result list, no full page chrome.
    payload = run_search(request, q, dim, top_k)
    return templates.TemplateResponse(
        request,
        "_results.html",
        {**payload, "corpus_size": len(request.app.state.corpus)},
    )


@app.get("/browse", response_class=HTMLResponse)
async def browse(request: Request):
    counts = Counter(doc["category"] for doc in request.app.state.corpus)
    # Sort by count descending so the busiest categories surface first.
    categories = sorted(
        ({"name": name, "label_ar": label_ar(name), "count": n} for name, n in counts.items()),
        key=lambda c: -c["count"],
    )
    return templates.TemplateResponse(
        request, "browse.html", {"categories": categories},
    )


@app.get("/browse/{category}", response_class=HTMLResponse)
async def browse_category(request: Request, category: str):
    entries = [doc for doc in request.app.state.corpus if doc["category"] == category]
    return templates.TemplateResponse(
        request,
        "category.html",
        {"category": category, "label_ar": label_ar(category), "entries": entries},
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(
        request, "about.html", {"corpus_size": len(request.app.state.corpus)},
    )
