from __future__ import annotations

import tempfile
import unittest

import numpy as np
from fastapi.testclient import TestClient

from app import create_app
from documents import MAX_UPLOAD_BYTES


class CountingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = np.ones((len(texts), 1024), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0] += len(text) / 100
        return vectors


class AppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.model = CountingModel()
        self.app = create_app(self.directory.name, model=self.model, seed_demo=False)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.token = self.app.state.csrf_token

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.directory.cleanup()

    def upload(self, text="توضح الوثيقة شروط القبول والتسجيل في الجامعة.", name="دليل.txt"):
        return self.client.post("/documents", data={"csrf_token": self.token},
                                files={"file": (name, text.encode(), "text/plain")})

    def test_pages_render_without_model_or_documents(self):
        for path in ("/", "/search", "/documents", "/browse", "/about", "/health"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
        self.assertEqual(self.model.calls, [])
        response = self.client.get("/api/query", params={"q": "شروط"})
        self.assertEqual(response.json()["status"], "no-index")

    def test_upload_search_source_replace_delete(self):
        self.assertEqual(self.upload().status_code, 200)
        document_id = self.app.state.store.documents()[0]["id"]
        data = self.client.get("/api/query", params={"q": "القبول", "method": "keyword"}).json()
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(self.model.calls, [])
        passage_id = data["results"][0]["id"]
        self.assertEqual(self.client.get(f"/passages/{passage_id}").status_code, 200)
        source = self.client.get(f"/documents/{document_id}/source")
        self.assertIn("شروط القبول", source.text)
        self.assertIn("text/plain", source.headers["content-type"])
        self.assertEqual(source.headers["content-security-policy"], "sandbox")
        replace = self.client.post(f"/documents/{document_id}/replace", data={"csrf_token": self.token},
                                   files={"file": ("new.txt", "دليل الصحة والنوم".encode(), "text/plain")})
        self.assertEqual(replace.status_code, 200)
        self.assertEqual(self.client.get(f"/passages/{passage_id}").status_code, 404)
        self.assertEqual(self.client.get("/api/query", params={"q": "القبول", "method": "keyword"}).json()["results"], [])
        self.assertEqual(self.client.post(f"/documents/{document_id}/delete", data={"csrf_token": self.token}).status_code, 200)
        self.assertEqual(self.app.state.store.documents(), [])
        self.assertEqual(self.client.get(f"/documents/{document_id}/source").status_code, 404)

    def test_dimension_change_reuses_query_encoding(self):
        self.upload()
        first = self.client.get("/api/query", params={"q": "الشروط", "dim": 1024}).json()
        calls = len(self.model.calls)
        second = self.client.get("/api/query", params={"q": "الشروط", "dim": 64}).json()
        self.assertEqual(first["status"], "results")
        self.assertFalse(first["query_cached"])
        self.assertTrue(second["query_cached"])
        self.assertEqual(calls, len(self.model.calls))

    def test_csrf_and_host_checks(self):
        response = self.client.post("/documents", data={"csrf_token": "wrong"},
                                    files={"file": ("a.txt", b"hello", "text/plain")})
        self.assertEqual(response.status_code, 403)
        response = self.client.post("/documents", data={"csrf_token": self.token},
                                    files={"file": ("a.txt", b"hello", "text/plain")},
                                    headers={"Origin": "https://attacker.example"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/", headers={"Host": "attacker.example"}).status_code, 400)

    def test_invalid_inputs_and_xss_are_handled(self):
        self.assertEqual(self.client.get("/api/query", params={"q": "a" * 2001}).status_code, 422)
        self.assertEqual(self.client.get("/api/query", params={"q": "a", "dim": 63}).status_code, 422)
        self.assertEqual(self.client.get("/api/query", params={"q": "a", "method": "magic"}).status_code, 422)
        self.upload('<script>alert("x")</script> University', "<script>.txt")
        response = self.client.get("/search", params={"q": "University", "method": "keyword"})
        self.assertNotIn('<script>alert("x")</script>', response.text)
        self.assertIn("&lt;script&gt;", response.text)

    def test_upload_size_and_feedback(self):
        response = self.client.post("/documents", data={"csrf_token": self.token},
                                    files={"file": ("large.txt", b"a" * (MAX_UPLOAD_BYTES + 1), "text/plain")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.app.state.store.documents(), [])
        self.upload()
        passage_id = self.app.state.store.passages()[0]["id"]
        document_id = self.app.state.store.documents()[0]["id"]
        response = self.client.post("/feedback", data={"csrf_token": self.token, "passage_id": passage_id,
            "query": "القبول", "helpful": "yes", "method": "keyword", "dim": 1024,
            "document_id": document_id, "top_k": 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.url.params["document_id"], document_id)
        self.assertEqual(response.url.params["top_k"], "3")
        with self.app.state.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
