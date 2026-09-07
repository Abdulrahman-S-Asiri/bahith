from __future__ import annotations

import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import create_app


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = np.ones((len(texts), 1024), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, 0] += (len(text) + index) / 100
        return vectors


class AppIntegrationReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.model = RecordingModel()
        self.app = create_app(self.directory.name, model=self.model, seed_demo=False)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.token = self.app.state.csrf_token

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self.directory.cleanup()

    def upload(self, text: str, filename: str = "document.txt"):
        return self.client.post(
            "/documents",
            data={"csrf_token": self.token},
            files={"file": (filename, text.encode(), "text/plain")},
            headers={"Origin": "http://testserver"},
        )

    def test_same_origin_forms_work_without_allowing_null_origin(self) -> None:
        page = self.client.get("/documents")
        accepted = self.upload("same origin content")
        rejected = self.client.post(
            "/documents",
            data={"csrf_token": self.token},
            files={"file": ("null.txt", b"must not import", "text/plain")},
            headers={"Origin": "null"},
        )

        self.assertEqual(page.headers["referrer-policy"], "same-origin")
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(len(self.app.state.store.documents()), 1)

    def test_mutation_discards_query_cache_and_removed_passage_state(self) -> None:
        secret = "PRIVATE_SENTINEL_4f882 document"
        self.upload(secret, "private.txt")
        document_id = self.app.state.store.documents()[0]["id"]
        first = self.client.get("/api/query", params={"q": "PRIVATE_SENTINEL_4f882"}).json()
        indexed_searcher = self.app.state.searcher
        self.assertFalse(first["query_cached"])

        deleted = self.client.post(
            f"/documents/{document_id}/delete",
            data={"csrf_token": self.token},
            headers={"Origin": "http://testserver"},
        )

        self.assertEqual(deleted.status_code, 200)
        self.assertIsNot(self.app.state.searcher, indexed_searcher)
        self.assertTrue(self.client.get("/health").json()["model_loaded"])
        if self.app.state.searcher is not None:
            self.assertFalse(any(secret in row["text"] for row in self.app.state.searcher.corpus))

    def test_deletion_clears_complete_embedding_cache_directory(self) -> None:
        self.upload("private cache source")
        document_id = self.app.state.store.documents()[0]["id"]
        cache = Path(self.directory.name) / "embeddings"
        cache.mkdir()
        (cache / "complete.npy").write_bytes(b"private vector")
        (cache / "interrupted-write.tmp").write_bytes(b"private partial vector")

        response = self.client.post(
            f"/documents/{document_id}/delete",
            data={"csrf_token": self.token},
            headers={"Origin": "http://testserver"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(cache.iterdir()), [])

    def test_cache_purge_failure_preserves_source_and_allows_retry(self) -> None:
        source = "source must survive a locked cache file"
        self.upload(source, "preserved.txt")
        document_id = self.app.state.store.documents()[0]["id"]
        cache = Path(self.directory.name) / "embeddings"
        cache.mkdir()
        locked = cache / "locked.npy"
        locked.write_bytes(b"locked vector")

        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            failed = self.client.post(
                f"/documents/{document_id}/delete",
                data={"csrf_token": self.token},
                headers={"Origin": "http://testserver"},
            )

        self.assertEqual(failed.status_code, 409)
        self.assertEqual(self.app.state.store.document(document_id, include_original=True)["original"], source.encode())
        self.assertEqual(self.client.get(f"/documents/{document_id}/source").text, source)
        self.assertTrue(locked.exists())

        retried = self.client.post(
            f"/documents/{document_id}/delete",
            data={"csrf_token": self.token},
            headers={"Origin": "http://testserver"},
        )
        self.assertEqual(retried.status_code, 200)
        self.assertIsNone(self.app.state.store.document(document_id))
        self.assertFalse(locked.exists())

    def test_replace_invalidates_old_passage_link_and_feedback(self) -> None:
        self.upload("original passage", "original.txt")
        document_id = self.app.state.store.documents()[0]["id"]
        old_passage = self.app.state.store.passages(document_id)[0]["id"]
        feedback = self.client.post(
            "/feedback",
            data={
                "csrf_token": self.token,
                "passage_id": old_passage,
                "query": "original",
                "helpful": "yes",
                "method": "keyword",
                "dim": 1024,
            },
            headers={"Origin": "http://testserver"},
        )
        self.assertEqual(feedback.status_code, 200)

        replacement = self.client.post(
            f"/documents/{document_id}/replace",
            data={"csrf_token": self.token},
            files={"file": ("replacement.txt", b"replacement passage", "text/plain")},
            headers={"Origin": "http://testserver"},
        )

        self.assertEqual(replacement.status_code, 200)
        self.assertEqual(self.app.state.store.documents()[0]["id"], document_id)
        self.assertEqual(self.client.get(f"/passages/{old_passage}").status_code, 404)
        with self.app.state.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 0)

    def test_passage_disappearing_during_snapshot_returns_not_found(self) -> None:
        self.upload("passage removed during request")
        store = self.app.state.store
        document_id = store.documents()[0]["id"]
        passage_id = store.passages(document_id)[0]["id"]
        read_passages = store.passages
        raced = False

        def delete_before_neighbor_read(requested_document_id: str | None = None):
            nonlocal raced
            if requested_document_id == document_id and not raced:
                raced = True
                store.delete(document_id)
            return read_passages(requested_document_id)

        with patch.object(store, "passages", side_effect=delete_before_neighbor_read):
            response = self.client.get(f"/passages/{passage_id}")

        self.assertEqual(response.status_code, 404)

    def test_delete_waits_for_passage_snapshot_lock(self) -> None:
        self.upload("stable passage snapshot")
        store = self.app.state.store
        document_id = store.documents()[0]["id"]
        passage_id = store.passages(document_id)[0]["id"]
        read_passages = store.passages
        snapshot_started = threading.Event()
        release_snapshot = threading.Event()

        def hold_snapshot(requested_document_id: str | None = None):
            rows = read_passages(requested_document_id)
            if requested_document_id == document_id:
                snapshot_started.set()
                self.assertTrue(release_snapshot.wait(timeout=2))
            return rows

        def delete_document():
            return self.client.post(
                f"/documents/{document_id}/delete",
                data={"csrf_token": self.token},
                headers={"Origin": "http://testserver"},
            )

        with patch.object(store, "passages", side_effect=hold_snapshot):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                snapshot_response = executor.submit(self.client.get, f"/passages/{passage_id}")
                self.assertTrue(snapshot_started.wait(timeout=2))
                delete_response = executor.submit(delete_document)
                threading.Event().wait(0.1)
                self.assertFalse(delete_response.done())
                release_snapshot.set()
                shown = snapshot_response.result(timeout=2)
                deleted = delete_response.result(timeout=2)

        self.assertEqual(shown.status_code, 200)
        self.assertEqual(deleted.status_code, 200)
        self.assertIsNone(store.document(document_id))

    def test_model_failure_does_not_expose_exception_details(self) -> None:
        class FailingModel(RecordingModel):
            def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
                raise RuntimeError("C:/private/path PRIVATE_SENTINEL_91d2")

        self.client.__exit__(None, None, None)
        self.app = create_app(self.directory.name, model=FailingModel(), seed_demo=False)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.token = self.app.state.csrf_token
        self.upload("searchable text")

        response = self.client.get("/api/query", params={"q": "searchable"})

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("PRIVATE_SENTINEL_91d2", response.text)
        self.assertNotIn("C:/private/path", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")


if __name__ == "__main__":
    unittest.main()
