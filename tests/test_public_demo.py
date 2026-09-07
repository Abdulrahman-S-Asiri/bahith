from __future__ import annotations

import concurrent.futures
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import create_app
from documents import DocumentStore


PUBLIC_ENV = {
    "BAHITH_PUBLIC_DEMO": "1",
    "BAHITH_PRELOAD_MODEL": "0",
    "BAHITH_ALLOWED_HOSTS": "demo.example",
    "BAHITH_ALLOW_HF_EMBED": "0",
}


class FastModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.ready = threading.Event()

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(list(texts))
        if len(self.calls) >= 2:
            self.ready.set()
        return np.ones((len(texts), 1024), dtype=np.float32)


class BlockingModel(FastModel):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.entered.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("test did not release model")
        return super().encode(texts, **kwargs)


class PublicDemoTests(unittest.TestCase):
    def test_public_mode_uses_an_isolated_demo_store_and_rejects_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as private_directory:
            private_store = DocumentStore(private_directory)
            private_id, _, _ = private_store.import_document("private.txt", b"PRIVATE_SENTINEL_7f1c")
            app = create_app(private_directory, model=FastModel(), seed_demo=False, public_demo=True)

            with TestClient(app) as client:
                public_path = Path(app.state.store.path)
                generation = app.state.store.generation
                documents = app.state.store.documents()
                self.assertNotEqual(public_path.parent, Path(private_directory))
                self.assertEqual([(row["id"], row["is_demo"]) for row in documents], [("demo", 1)])
                self.assertFalse(any("PRIVATE_SENTINEL_7f1c" in row["text"] for row in app.state.store.passages()))
                self.assertIsNone(app.state.csrf_token)
                self.assertNotIn('name="csrf_token"', client.get("/documents").text)
                self.assertNotIn("PHISHING_SENTINEL", client.get("/documents?message=PHISHING_SENTINEL").text)
                self.assertEqual(client.get(f"/documents/{private_id}/source").status_code, 404)

                for path in ("/documents", "/documents/demo/replace", "/documents/demo/delete", "/feedback"):
                    response = client.post(path, content=b"body-must-not-be-parsed")
                    self.assertEqual(response.status_code, 405, path)
                    self.assertEqual(response.headers["allow"], "GET, HEAD")
                self.assertEqual(app.state.store.generation, generation)
                with app.state.store.connection() as db:
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 0)

            self.assertFalse(public_path.exists())
            self.assertEqual(private_store.document(private_id, include_original=True)["original"], b"PRIVATE_SENTINEL_7f1c")

    def test_env_enables_public_mode_and_exact_host_allowlist(self) -> None:
        with patch.dict(os.environ, PUBLIC_ENV, clear=False):
            app = create_app(model=FastModel())
            with TestClient(app) as client:
                self.assertTrue(app.state.public_demo)
                self.assertEqual(client.get("/", headers={"Host": "demo.example"}).status_code, 200)
                self.assertEqual(client.get("/", headers={"Host": "attacker.example"}).status_code, 400)
                self.assertEqual(client.get("/docs").status_code, 404)
                self.assertEqual(client.get("/openapi.json").status_code, 404)

        with patch.dict(os.environ, {"BAHITH_ALLOWED_HOSTS": "*"}, clear=False):
            with self.assertRaisesRegex(ValueError, "exact host names"):
                create_app(public_demo=True)

    def test_public_query_limits_run_before_the_model(self) -> None:
        model = FastModel()
        app = create_app(model=model, public_demo=True)
        with TestClient(app) as client:
            too_many_characters = client.get("/api/query", params={"q": "a" * 2001})
            too_many_url_bytes = client.get("/about?unused=" + "a" * 24_001)

        self.assertEqual(too_many_characters.status_code, 422)
        self.assertEqual(too_many_url_bytes.status_code, 414)
        self.assertEqual(model.calls, [])

    def test_public_search_gate_rejects_parallel_work_without_queueing(self) -> None:
        model = BlockingModel()
        app = create_app(model=model, public_demo=True)
        with TestClient(app) as client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(client.get, "/api/query", params={"q": "السؤال الأول"})
                self.assertTrue(model.entered.wait(timeout=2))
                busy = client.get("/api/query", params={"q": "السؤال الثاني"})
                self.assertEqual(busy.status_code, 429)
                self.assertEqual(busy.headers["retry-after"], "2")
                model.release.set()
                self.assertEqual(first.result(timeout=3).status_code, 200)

            available = client.get("/api/query", params={"q": "القراءة", "method": "keyword"})
            self.assertEqual(available.status_code, 200)

    def test_readiness_requires_a_prepared_dense_model_and_index(self) -> None:
        model = FastModel()
        app = create_app(model=model, public_demo=True)
        with TestClient(app) as client:
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/api/query", params={"q": "القراءة", "method": "keyword"}).status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/api/query", params={"q": "القراءة"}).status_code, 200)
            self.assertEqual(client.get("/ready").json(), {"status": "ready"})

    def test_document_mutation_resets_dense_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory, model=FastModel(), seed_demo=False, public_demo=False)
            with TestClient(app) as client:
                token = app.state.csrf_token
                first_upload = client.post("/documents", data={"csrf_token": token},
                    files={"file": ("first.txt", "نص عربي للبحث".encode(), "text/plain")})
                self.assertEqual(first_upload.status_code, 200)
                self.assertEqual(client.get("/api/query", params={"q": "البحث"}).status_code, 200)
                self.assertEqual(client.get("/ready").status_code, 200)

                second_upload = client.post("/documents", data={"csrf_token": token},
                    files={"file": ("second.txt", "مصدر عربي آخر".encode(), "text/plain")})
                self.assertEqual(second_upload.status_code, 200)
                self.assertEqual(client.get("/ready").status_code, 503)
                self.assertFalse(app.state.readiness_event.is_set())

    def test_background_preload_holds_capacity_until_ready(self) -> None:
        model = FastModel()
        with patch.dict(os.environ, {**PUBLIC_ENV, "BAHITH_PRELOAD_MODEL": "1"}, clear=False):
            app = create_app(model=model)
            with TestClient(app) as client:
                self.assertTrue(model.ready.wait(timeout=2))
                self.assertTrue(app.state.readiness_event.wait(timeout=2))
                self.assertEqual(client.get("/ready").json(), {"status": "ready"})

    def test_public_security_headers_default_to_no_embedding(self) -> None:
        app = create_app(model=FastModel(), public_demo=True)
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.headers["x-frame-options"], "DENY")
            self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
            self.assertEqual(response.headers["permissions-policy"], "camera=(), microphone=(), geolocation=()")


if __name__ == "__main__":
    unittest.main()
