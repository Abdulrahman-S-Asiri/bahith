from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from documents import (
    MAX_PAGES,
    MAX_TEXT_CHARS,
    MAX_UPLOAD_BYTES,
    DocumentError,
    DocumentStore,
    chunk_pages,
    extract_pages,
)


def make_pdf(pages: list[str | None], password: str | None = None) -> bytes:
    """Create the smallest useful in-memory PDFs for extraction and failure tests."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        if text is not None:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 20 260 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
            )
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class DocumentExtractionTests(unittest.TestCase):
    def test_chunking_covers_unicode_text_and_keeps_page_references(self) -> None:
        first = "".join(chr(0x4E00 + index) for index in range(430))
        second = "".join(chr(0x5200 + index) for index in range(275))
        pages = [(2, first), (9, second)]

        chunks = chunk_pages(pages, size=120, overlap=30)

        self.assertEqual([chunk["ordinal"] for chunk in chunks], list(range(1, len(chunks) + 1)))
        for page_number, source in pages:
            page_chunks = [chunk for chunk in chunks if chunk["page"] == page_number]
            self.assertTrue(page_chunks)
            covered = [False] * len(source)
            previous_start = -1
            for chunk in page_chunks:
                start = source.find(chunk["text"], previous_start + 1)
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start, previous_start + 120)
                for index in range(start, start + len(chunk["text"])):
                    covered[index] = True
                previous_start = start
            self.assertTrue(all(covered))

    def test_rejects_malformed_encrypted_and_blank_pdfs_safely(self) -> None:
        fixtures = {
            "malformed": b"%PDF-1.7\nnot a complete pdf",
            "encrypted": make_pdf(["secret"], password="password"),
            "blank": make_pdf([None]),
        }
        for label, content in fixtures.items():
            with self.subTest(label=label):
                with self.assertRaises(DocumentError):
                    extract_pages(f"{label}.pdf", content)

    def test_partial_blank_pdf_imports_text_and_returns_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(directory)

            document_id, created, warning = store.import_document(
                "partial.pdf", make_pdf(["searchable page", None])
            )

            self.assertTrue(created)
            self.assertIn("1", warning or "")
            self.assertEqual(store.document(document_id)["page_count"], 2)  # type: ignore[index]
            passages = store.passages(document_id)
            self.assertEqual([(row["page"], row["text"]) for row in passages], [(1, "searchable page")])

    def test_upload_size_text_size_and_page_limits(self) -> None:
        cases = [
            ("too-large.txt", b"x" * (MAX_UPLOAD_BYTES + 1)),
            ("too-much-text.txt", b"x" * (MAX_TEXT_CHARS + 1)),
            ("too-many-pages.pdf", make_pdf([None] * (MAX_PAGES + 1))),
        ]
        for filename, content in cases:
            with self.subTest(filename=filename):
                with self.assertRaises(DocumentError):
                    extract_pages(filename, content)


class DocumentStoreTests(unittest.TestCase):
    def test_persistence_roundtrip_preserves_original_and_passages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            content = "السطر الأول\n\nالسطر الثاني".encode("utf-8")
            first = DocumentStore(directory)
            document_id, created, warning = first.import_document("بحث.md", content)
            generation = first.generation

            reopened = DocumentStore(directory)
            stored = reopened.document(document_id, include_original=True)

            self.assertTrue(created)
            self.assertIsNone(warning)
            self.assertEqual(reopened.generation, generation)
            self.assertEqual(stored["filename"], "بحث.md")  # type: ignore[index]
            self.assertEqual(stored["title"], "بحث")  # type: ignore[index]
            self.assertEqual(stored["original"], content)  # type: ignore[index]
            self.assertEqual(
                [row["text"] for row in reopened.passages(document_id)],
                ["السطر الأول\n\nالسطر الثاني"],
            )

    def test_exact_duplicate_returns_existing_document_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(directory)
            content = "identical bytes".encode()
            first_id, created, _ = store.import_document("first.txt", content)
            generation = store.generation

            duplicate_id, duplicate_created, warning = store.import_document("alias.md", content)

            self.assertTrue(created)
            self.assertEqual(duplicate_id, first_id)
            self.assertFalse(duplicate_created)
            self.assertIsNone(warning)
            self.assertEqual(store.generation, generation)
            self.assertEqual(len(store.documents()), 1)
            self.assertEqual(store.document(first_id)["filename"], "first.txt")  # type: ignore[index]

    def test_replacement_rolls_back_if_insert_fails_after_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(directory)
            original = b"original searchable text"
            document_id, _, _ = store.import_document("original.txt", original)
            old_passages = store.passages(document_id)
            old_generation = store.generation
            with store.connection() as db:
                db.executescript("""
                    CREATE TRIGGER reject_test_replacement
                    BEFORE INSERT ON documents
                    WHEN NEW.filename = 'replacement.txt'
                    BEGIN
                        SELECT RAISE(ABORT, 'forced replacement failure');
                    END;
                """)

            with self.assertRaises(sqlite3.IntegrityError):
                store.import_document("replacement.txt", b"new searchable text", replace_id=document_id)

            preserved = store.document(document_id, include_original=True)
            self.assertEqual(preserved["original"], original)  # type: ignore[index]
            self.assertEqual(preserved["filename"], "original.txt")  # type: ignore[index]
            self.assertEqual(store.passages(document_id), old_passages)
            self.assertEqual(store.generation, old_generation)

    def test_deleted_demo_document_does_not_reappear_on_reopen(self) -> None:
        demo = [{"id": 41, "text": "مقطع تجريبي", "category": "demo"}]
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(directory, demo_corpus=demo)
            self.assertTrue(store.delete("demo"))

            reopened = DocumentStore(directory, demo_corpus=demo)

            self.assertEqual(reopened.documents(), [])
            self.assertEqual(reopened.passages(), [])
            self.assertFalse(reopened.delete("demo"))

    def test_import_sanitizes_path_components_from_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(directory)

            document_id, _, _ = store.import_document("../../private\\safe.txt", b"safe content")
            stored = store.document(document_id)

            self.assertEqual(stored["filename"], "safe.txt")  # type: ignore[index]
            self.assertEqual(stored["title"], "safe")  # type: ignore[index]
            self.assertEqual(list(Path(directory).iterdir()), [Path(directory) / "library.sqlite3"])


if __name__ == "__main__":
    unittest.main()
