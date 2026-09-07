from __future__ import annotations

import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy.export_pages import export_pages, normalize_base_path, validate_api_base


class PagesExportTests(unittest.TestCase):
    def test_exports_isolated_seeded_site_without_loading_a_model(self) -> None:
        imported = builtins.__import__

        def reject_model_import(name, *args, **kwargs):
            if name == "sentence_transformers" or name.startswith("sentence_transformers."):
                raise AssertionError("the static export must not import the embedding backend")
            return imported(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            with patch("builtins.__import__", side_effect=reject_model_import):
                written = export_pages(output, base_path="/bahith/", api_base="")

            expected_pages = {
                output / "index.html",
                output / "search" / "index.html",
                output / "documents" / "index.html",
                output / "documents" / "demo" / "index.html",
                output / "documents" / "demo" / "source" / "index.html",
                output / "browse" / "index.html",
                output / "about" / "index.html",
                output / "404.html",
            }
            expected_pages.update(output / "passages" / str(value) / "index.html" for value in range(1, 31))
            for category in ("religion", "health", "tech", "science", "poetry", "economy",
                             "sports", "education", "culture"):
                expected_pages.add(output / "browse" / category / "index.html")

            self.assertEqual(set(written), expected_pages)
            self.assertTrue(all(path.is_file() for path in expected_pages))
            self.assertTrue((output / ".nojekyll").is_file())
            self.assertTrue((output / "LICENSE").is_file())
            self.assertTrue((output / "static" / "css" / "style.css").is_file())
            self.assertTrue((output / "static" / "js" / "pages.js").is_file())
            self.assertFalse((output / "corpus.json").exists())
            self.assertFalse((output / ".git").exists())
            self.assertFalse(any(output.rglob("*.sqlite3")))

    def test_rewrites_nested_navigation_and_static_search_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            export_pages(output, base_path="bahith", api_base="https://search.example:8443/")

            search = (output / "search" / "index.html").read_text(encoding="utf-8")
            passage = (output / "passages" / "1" / "index.html").read_text(encoding="utf-8")
            source = (output / "documents" / "demo" / "source" / "index.html").read_text(encoding="utf-8")
            missing = (output / "404.html").read_text(encoding="utf-8")

            for page in (search, passage, source, missing):
                self.assertIn('<meta name="bahith-base-path" content="/bahith/">', page)
                self.assertIn('<meta name="bahith-api-base" content="https://search.example:8443">', page)
                self.assertNotRegex(page, r'(?:href|src|action)=["\']/(?!bahith/)')
            self.assertIn('src="/bahith/static/js/pages.js"', search)
            self.assertNotIn("app.js", search)
            self.assertIn('action="/bahith/search/"', search)
            self.assertIn("يلزم JavaScript للبحث المباشر", search)
            self.assertIn("ابدأ بسؤال", search)
            self.assertNotIn("أقرب المقاطع إلى سؤالك", search)
            self.assertNotRegex(search, r'<form\b[^>]*method=["\']post["\']')
            self.assertIn('href="/bahith/documents/demo/source/#page=1"', passage)
            self.assertIn('href="/bahith/search/?document_id=demo"', passage)
            self.assertIn("&quot;category&quot;", source)
            self.assertIn('href="/bahith/"', missing)
            self.assertIn('href="/bahith/browse/"', missing)

    def test_rejects_unsafe_configuration_and_nonempty_output(self) -> None:
        self.assertEqual(normalize_base_path("bahith"), "/bahith/")
        self.assertEqual(normalize_base_path("/owner/site/"), "/owner/site/")
        for value in ("", "/", "../site", "/site/../other", "/%2e%2e/site", "//evil/site",
                      "https://example.test/site", "/site?q=x", '/site"x'):
            with self.subTest(base_path=value):
                with self.assertRaises(ValueError):
                    normalize_base_path(value)

        self.assertEqual(validate_api_base(""), "")
        self.assertEqual(validate_api_base("https://api.example/"), "https://api.example")
        for value in ("http://api.example", "//api.example", "https://api.example/v1",
                      "https://api.example?q=x", "https://user@api.example", "https://exa mple.test"):
            with self.subTest(api_base=value):
                with self.assertRaises(ValueError):
                    validate_api_base(value)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_pages(output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "do not overwrite")
            self.assertEqual(list(output.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
