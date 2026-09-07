"""Local Arabic document search. All mutations require a same-origin CSRF token."""
from __future__ import annotations

import os
import secrets
import tempfile
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from documents import MAX_UPLOAD_BYTES, DocumentError, DocumentStore
from public_demo import (MAX_QUERY_BYTES, MAX_QUERY_STRING_BYTES, SEARCH_PATHS, allowed_hosts,
                         env_flag)
from search import (DEFAULT_DIM, DEFAULT_TOP_K, MAX_QUERY_CHARS, MAX_TOP_K, METHODS,
                    MODEL_NAME, MODEL_REVISION, SUPPORTED_DIMS, ArabicSearcher, load_corpus)

ROOT = Path(__file__).parent
CATEGORY_LABELS_AR = {"religion": "دين", "health": "صحة", "tech": "تقنية", "science": "علوم",
                      "poetry": "شعر", "economy": "اقتصاد", "sports": "رياضة", "education": "تعليم",
                      "culture": "ثقافة", "documents": "مستنداتي"}
SUGGESTIONS = ["ما فوائد القراءة؟", "كيف يتعلم الذكاء الاصطناعي من البيانات؟", "علاقة جودة النوم بالتركيز"]


def create_app(data_dir: str | Path | None = None, *, model=None, seed_demo: bool = True,
               public_demo: bool | None = None) -> FastAPI:
    public_demo = env_flag("BAHITH_PUBLIC_DEMO") if public_demo is None else public_demo
    preload_model = env_flag("BAHITH_PRELOAD_MODEL")
    allow_hf_embed = public_demo and env_flag("BAHITH_ALLOW_HF_EMBED")
    configured_directory = Path(data_dir or os.getenv("BAHITH_DATA_DIR", str(ROOT / ".bahith")))
    search_gate = threading.BoundedSemaphore(1)
    templates = Jinja2Templates(directory=ROOT / "templates")
    templates.env.globals["label_ar"] = lambda category: CATEGORY_LABELS_AR.get(category, category)

    def cleanup_public_directory(state) -> None:
        with state.cleanup_lock:
            temporary_directory = state.temporary_directory
            if temporary_directory is not None:
                state.temporary_directory = None
                temporary_directory.cleanup()

    @asynccontextmanager
    async def lifespan(application):
        temporary_directory = tempfile.TemporaryDirectory(prefix="bahith-public-") if public_demo else None
        directory = Path(temporary_directory.name) if temporary_directory else configured_directory
        application.state.data_directory = directory
        application.state.store = DocumentStore(directory, load_corpus() if seed_demo or public_demo else None)
        application.state.csrf_token = None if public_demo else secrets.token_urlsafe(32)
        application.state.lock = threading.RLock()
        application.state.public_demo = public_demo
        application.state.ready = False
        application.state.readiness_error = None
        application.state.readiness_event = threading.Event()
        application.state.shutdown_requested = threading.Event()
        application.state.cleanup_lock = threading.Lock()
        application.state.temporary_directory = temporary_directory
        application.state.searcher = None
        application.state.model = model
        application.state.generation = -1
        warm_thread = None
        if preload_model:
            search_gate.acquire()
            warm_thread = threading.Thread(target=warm_search, args=(application,), daemon=True,
                                           name="bahith-model-warmup")
            warm_thread.start()
        try:
            yield
        finally:
            application.state.shutdown_requested.set()
            if warm_thread:
                warm_thread.join(timeout=5)
            if not (warm_thread and warm_thread.is_alive()):
                application.state.searcher = None
                application.state.model = None
                cleanup_public_directory(application.state)

    application = FastAPI(title="باحث | Bahith", lifespan=lifespan,
                          docs_url=None if public_demo else "/docs",
                          redoc_url=None if public_demo else "/redoc",
                          openapi_url=None if public_demo else "/openapi.json")
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts())
    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @application.middleware("http")
    async def response_headers(request: Request, call_next):
        def secure(response: Response) -> Response:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "same-origin"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            frame_ancestors = "https://huggingface.co https://hf.space https://*.hf.space" if allow_hf_embed else "'none'"
            response.headers.setdefault("Content-Security-Policy", "default-src 'self'; base-uri 'none'; "
                "object-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                f"connect-src 'self'; form-action 'self'; frame-ancestors {frame_ancestors}")
            if allow_hf_embed:
                response.headers.pop("X-Frame-Options", None)
            else:
                response.headers["X-Frame-Options"] = "DENY"
            if not request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-store"
            return response

        if public_demo and request.method not in ("GET", "HEAD"):
            return secure(JSONResponse({"detail": "العرض العام للقراءة والبحث فقط."}, status_code=405,
                                       headers={"Allow": "GET, HEAD"}))

        acquired = False
        if public_demo and len(request.scope.get("query_string", b"")) > MAX_QUERY_STRING_BYTES:
            return secure(JSONResponse({"detail": "رابط الطلب طويل جدًا."}, status_code=414))
        if public_demo and request.url.path in SEARCH_PATHS:
            query = request.query_params.get("q", "")
            if len(query.strip()) > MAX_QUERY_CHARS or len(query.encode("utf-8")) > MAX_QUERY_BYTES:
                return secure(JSONResponse({"detail": "الاستعلام طويل جدًا؛ الحد الأقصى 2000 حرف."}, status_code=422))
            if query.strip():
                acquired = search_gate.acquire(blocking=False)
                if not acquired:
                    return secure(JSONResponse({"detail": "خدمة البحث مشغولة الآن؛ أعد المحاولة بعد لحظات."},
                                               status_code=429, headers={"Retry-After": "2"}))
        try:
            response = await call_next(request)
        finally:
            if acquired:
                search_gate.release()
        return secure(response)

    def render(request: Request, template: str, context: dict | None = None, status_code: int = 200):
        base = {"csrf_token": request.app.state.csrf_token, "public_demo": request.app.state.public_demo,
                "documents": request.app.state.store.documents(),
                "model_name": MODEL_NAME, "model_revision": MODEL_REVISION}
        return templates.TemplateResponse(request=request, name=template, context={**base, **(context or {})}, status_code=status_code)

    def csrf(request: Request, token: str):
        origin = request.headers.get("origin")
        if origin and (origin == "null" or urlsplit(origin).netloc != request.url.netloc):
            raise HTTPException(403, "مصدر الطلب غير مسموح.")
        if not secrets.compare_digest(token, request.app.state.csrf_token):
            raise HTTPException(403, "انتهت صلاحية النموذج؛ أعد تحميل الصفحة.")

    def state_searcher(state):
        generation = state.store.generation
        if state.searcher is None or state.generation != generation:
            old_model = state.model
            state.searcher = ArabicSearcher(state.store.passages(), model=old_model,
                                           cache_dir=state.data_directory / "embeddings", load_embeddings=False)
            # Reusing a real model object must still permit its pinned persistent cache.
            state.searcher._injected_model = model is not None
            state.generation = generation
        return state.searcher

    def searcher(request: Request):
        return state_searcher(request.app.state)

    def warm_search(application: FastAPI) -> None:
        state = application.state
        try:
            with state.lock:
                engine = state_searcher(state)
                engine.embeddings
                engine.encode_query(SUGGESTIONS[0])
                state.ready = True
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            state.readiness_error = type(exc).__name__
        finally:
            state.readiness_event.set()
            search_gate.release()
            if state.shutdown_requested.is_set():
                state.searcher = None
                state.model = None
                cleanup_public_directory(state)

    def invalidate(request: Request):
        state = request.app.state
        if state.searcher and state.searcher._model is not None:
            state.model = state.searcher._model
        state.searcher = None  # Drop passage text, embeddings and query cache immediately.
        state.generation = -1
        state.ready = False
        state.readiness_error = None
        state.readiness_event.clear()

    def purge_derived_cache():
        # Purge before a source mutation: failures leave the source intact and retryable.
        cache = configured_directory / "embeddings"
        try:
            if cache.is_symlink():
                raise OSError("cache directory must not be a symbolic link")
            if cache.is_dir():
                for path in cache.iterdir():
                    if path.is_file() or path.is_symlink():
                        path.unlink(missing_ok=True)
        except OSError as exc:
            raise DocumentError("تعذّر مسح الفهرس المؤقت؛ أغلق البرامج التي تستخدمه وأعد المحاولة. لم نغيّر المستند الأصلي.") from exc

    def run_search(request, q, dim, top_k, method, document_id):
        if dim not in SUPPORTED_DIMS or method not in METHODS or not 1 <= top_k <= MAX_TOP_K:
            raise HTTPException(422, "إعدادات البحث غير صالحة.")
        if len(q.strip()) > MAX_QUERY_CHARS:
            raise HTTPException(422, "الاستعلام طويل جدًا؛ الحد الأقصى 2000 حرف.")
        state = request.app.state
        started = time.perf_counter()
        payload = {"q": q.strip(), "results": [], "embedding": None, "elapsed_ms": 0.0,
                   "dim": dim, "top_k": top_k, "method": method, "document_id": document_id or "",
                   "supported_dims": SUPPORTED_DIMS, "query_cached": False, "status": "idle", "error": None}
        with state.lock:
            engine = searcher(request)
            payload["corpus_size"] = sum(not document_id or row.get("document_id") == document_id for row in engine.corpus)
            if document_id and state.store.document(document_id) is None:
                raise HTTPException(404, "المستند غير موجود.")
            if not q.strip():
                return payload
            if not engine.corpus:
                payload["status"] = "no-index"
                return payload
            try:
                payload["results"] = engine.retrieve(q, top_k, dim, method, document_id=document_id)
                payload["query_cached"] = engine.last_query_cached
                if method != "keyword" and payload["results"]:
                    vector = engine.encode_query(q)
                    payload["embedding"] = engine.explain_vector(vector, dim)
                    state.ready = True
                    state.readiness_event.set()
                payload["status"] = "results" if payload["results"] else "no-results"
            except (OSError, RuntimeError, ImportError, ValueError) as exc:
                # Never expose paths, model internals, document text or tracebacks through the page.
                payload["status"] = "error"
                payload["error"] = "تعذّر تشغيل النموذج. جرّب البحث بالكلمات أو تحقّق من تثبيت النموذج حسب دليل التشغيل."
                payload["error_type"] = type(exc).__name__
            payload["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return payload

    @application.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return render(request, "home.html", {"suggestions": SUGGESTIONS, "corpus_size": len(request.app.state.store.passages())})

    @application.get("/search", response_class=HTMLResponse)
    def search(request: Request, q: str = "", dim: int = DEFAULT_DIM, top_k: int = DEFAULT_TOP_K,
               method: str = "semantic", document_id: str = ""):
        return render(request, "search.html", run_search(request, q, dim, top_k, method, document_id))

    @application.get("/api/search", response_class=HTMLResponse)
    def api_search(request: Request, q: str = "", dim: int = DEFAULT_DIM, top_k: int = DEFAULT_TOP_K,
                   method: str = "semantic", document_id: str = ""):
        return render(request, "_results.html", run_search(request, q, dim, top_k, method, document_id))

    @application.get("/api/query")
    def api_query(request: Request, q: str = "", dim: int = DEFAULT_DIM, top_k: int = DEFAULT_TOP_K,
                  method: str = "semantic", document_id: str = ""):
        result = run_search(request, q, dim, top_k, method, document_id)
        if result["status"] == "error":
            raise HTTPException(503, result["error"])
        return result

    @application.get("/health")
    def health(request: Request):
        state = request.app.state
        return {"status": "ok", "model_loaded": state.model is not None or bool(state.searcher and state.searcher._model is not None)}

    @application.get("/ready")
    def ready(request: Request):
        if request.app.state.ready:
            return {"status": "ready"}
        raise HTTPException(503, "النموذج والفهرس لم يجهزا بعد.")

    @application.get("/documents", response_class=HTMLResponse)
    def documents(request: Request, message: str = "", error: str = ""):
        if request.app.state.public_demo:
            message = error = ""
        return render(request, "documents.html", {"message": message[:500], "error": error[:500]})

    def import_file(request: Request, file: UploadFile, token: str, replace_id: str | None = None):
        csrf(request, token)
        try:
            content = file.file.read(MAX_UPLOAD_BYTES + 1)
            with request.app.state.lock:
                if replace_id:
                    if request.app.state.store.document(replace_id) is None:
                        raise DocumentError("المستند المطلوب تحديثه غير موجود.")
                    purge_derived_cache()
                document_id, changed, warning = request.app.state.store.import_document(file.filename or "", content, replace_id)
                if changed:
                    invalidate(request)
            message = "تم حفظ المستند. تُجهّز تمثيلاته عند أول بحث دلالي." if changed else "هذا الملف موجود بالفعل؛ لم نضف نسخة مكررة."
            if warning:
                message += " " + warning
            return RedirectResponse("/documents?" + urlencode({"message": message}), status_code=303)
        except DocumentError as exc:
            return render(request, "documents.html", {"error": str(exc), "message": ""}, 400)
        finally:
            file.file.close()

    @application.post("/documents")
    def upload(request: Request, file: UploadFile = File(...), csrf_token: str = Form(...)):
        return import_file(request, file, csrf_token)

    @application.post("/documents/{document_id}/replace")
    def replace(request: Request, document_id: str, file: UploadFile = File(...), csrf_token: str = Form(...)):
        return import_file(request, file, csrf_token, document_id)

    @application.post("/documents/{document_id}/delete")
    def delete(request: Request, document_id: str, csrf_token: str = Form(...)):
        csrf(request, csrf_token)
        with request.app.state.lock:
            if request.app.state.store.document(document_id) is None:
                raise HTTPException(404, "المستند غير موجود.")
            try:
                purge_derived_cache()
            except DocumentError as exc:
                return render(request, "documents.html", {"error": str(exc)}, 409)
            request.app.state.store.delete(document_id)
            invalidate(request)
        return RedirectResponse("/documents?" + urlencode({"message": "تم حذف المستند ومقاطعه وتقييماته."}), status_code=303)

    @application.get("/documents/{document_id}", response_class=HTMLResponse)
    def document(request: Request, document_id: str):
        with request.app.state.lock:
            record = request.app.state.store.document(document_id)
            if record is None:
                raise HTTPException(404, "المستند غير موجود.")
            passages = request.app.state.store.passages(document_id)
        return render(request, "document.html", {"document": record, "passages": passages})

    @application.get("/documents/{document_id}/source")
    def source(request: Request, document_id: str):
        record = request.app.state.store.document(document_id, include_original=True)
        if record is None:
            raise HTTPException(404, "المستند غير موجود.")
        pdf = record["filename"].lower().endswith(".pdf")
        return Response(record["original"], media_type="application/pdf" if pdf else "text/plain; charset=utf-8",
                        headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(record['filename'], safe='')}",
                                 "Content-Security-Policy": "sandbox"})

    @application.get("/passages/{passage_id}", response_class=HTMLResponse)
    def passage(request: Request, passage_id: int):
        with request.app.state.lock:
            record = request.app.state.store.passage(passage_id)
            if record is None:
                raise HTTPException(404, "المقطع غير موجود.")
            neighbors = request.app.state.store.passages(record["document_id"])
            index = next((i for i, row in enumerate(neighbors) if row["id"] == passage_id), None)
            document = request.app.state.store.document(record["document_id"])
            if index is None or document is None:
                raise HTTPException(404, "المقطع غير موجود.")
        return render(request, "passage.html", {"passage": record, "document": document,
            "previous": neighbors[index - 1] if index else None,
            "next": neighbors[index + 1] if index + 1 < len(neighbors) else None})

    @application.post("/feedback")
    def feedback(request: Request, passage_id: int = Form(...), query: str = Form(...), helpful: str = Form(...),
                 method: str = Form("semantic"), dim: int = Form(DEFAULT_DIM), csrf_token: str = Form(...),
                 document_id: str = Form(""), top_k: int = Form(DEFAULT_TOP_K)):
        csrf(request, csrf_token)
        if (helpful not in ("yes", "no") or method not in METHODS or dim not in SUPPORTED_DIMS
                or len(query) > MAX_QUERY_CHARS or not 1 <= top_k <= MAX_TOP_K):
            raise HTTPException(422, "تقييم غير صالح.")
        with request.app.state.lock:
            if document_id and request.app.state.store.document(document_id) is None:
                raise HTTPException(404, "المستند غير موجود.")
            if request.app.state.store.passage(passage_id) is None:
                raise HTTPException(404, "المقطع غير موجود.")
            request.app.state.store.add_feedback(passage_id, query, helpful == "yes", method, dim)
        return RedirectResponse("/search?" + urlencode({"q": query, "dim": dim, "method": method,
                                "document_id": document_id, "top_k": top_k}), status_code=303)

    @application.get("/browse", response_class=HTMLResponse)
    def browse(request: Request):
        counts = Counter(row["category"] for row in request.app.state.store.passages())
        categories = [{"name": key, "label_ar": CATEGORY_LABELS_AR.get(key, key), "count": count}
                      for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
        return render(request, "browse.html", {"categories": categories})

    @application.get("/browse/{category}", response_class=HTMLResponse)
    def browse_category(request: Request, category: str):
        return render(request, "category.html", {"category": category, "label_ar": CATEGORY_LABELS_AR.get(category, category),
            "entries": [row for row in request.app.state.store.passages() if row["category"] == category]})

    @application.get("/about", response_class=HTMLResponse)
    def about(request: Request):
        return render(request, "about.html", {"corpus_size": len(request.app.state.store.passages())})

    return application


app = create_app()
