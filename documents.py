"""Local SQLite document library. Original uploads and passages share one transaction."""
from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGES = 300
MAX_TEXT_CHARS = 500_000
CHUNK_CHARS = 1000
OVERLAP_CHARS = 120
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentError(ValueError):
    pass


def extract_pages(filename: str, content: bytes) -> list[tuple[int, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise DocumentError("الملفات المدعومة: PDF نصي، TXT، Markdown.")
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise DocumentError("اختر ملفًا غير فارغ لا يتجاوز 10 ميجابايت.")
    if suffix == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise DocumentError("الملف مشفّر. أضف نسخة يمكن قراءتها دون كلمة مرور.")
            if len(reader.pages) > MAX_PAGES:
                raise DocumentError("الحد الأقصى 300 صفحة لكل ملف.")
            pages, total = [], 0
            for index, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                total += len(text)
                if total > MAX_TEXT_CHARS:
                    raise DocumentError("النص المستخرج كبير جدًا؛ قسّم المستند إلى ملفات أصغر.")
                pages.append((index, text))
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("تعذّر قراءة ملف PDF. تأكد من سلامته.") from exc
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError("احفظ الملف النصي بترميز UTF-8 ثم أعد المحاولة.") from exc
        if "\x00" in text:
            raise DocumentError("الملف لا يبدو ملفًا نصيًا صالحًا.")
        if len(text) > MAX_TEXT_CHARS:
            raise DocumentError("النص كبير جدًا؛ قسّمه إلى ملفات أصغر.")
        pages = [(1, text)]
    if not any(text.strip() for _, text in pages):
        raise DocumentError("لم يُستخرج نص قابل للبحث. ملفات PDF المصوّرة تحتاج OCR غير متاح في هذه النسخة.")
    return pages


def chunk_pages(pages: list[tuple[int, str]], size: int = CHUNK_CHARS,
                overlap: int = OVERLAP_CHARS) -> list[dict]:
    """Keep page boundaries; prefer paragraph/sentence/word breaks; never silently drop text."""
    if size < 100 or overlap < 0 or overlap >= size // 2:
        raise ValueError("invalid chunk size/overlap")
    chunks = []
    for page, original in pages:
        text = original.replace("\r\n", "\n").replace("\r", "\n").strip()
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                for separator in ("\n\n", "\n", ". ", "؟ ", "؛ ", " "):
                    split = text.rfind(separator, start + size // 2, end)
                    if split >= 0:
                        end = split + len(separator)
                        break
            passage = text[start:end].strip()
            if passage:
                chunks.append({"page": page, "ordinal": len(chunks) + 1, "text": passage})
            if end == len(text):
                break
            next_start = max(start + 1, end - overlap)
            # Move an overlapping prefix to a word boundary without leaving a gap.
            if next_start > start and text[next_start - 1:next_start] not in (" ", "\n"):
                boundary = text.find(" ", next_start, end)
                if boundary >= 0:
                    next_start = boundary + 1
            start = min(next_start, end)
    return chunks


class DocumentStore:
    def __init__(self, directory: str | Path, demo_corpus: list[dict] | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "library.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
                    sha256 TEXT UNIQUE NOT NULL, page_count INTEGER NOT NULL,
                    original BLOB NOT NULL, created_at TEXT NOT NULL, is_demo INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS passages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'documents', UNIQUE(document_id, ordinal));
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY, passage_id INTEGER REFERENCES passages(id) ON DELETE CASCADE,
                    query TEXT NOT NULL, helpful INTEGER NOT NULL, method TEXT NOT NULL,
                    dimension INTEGER NOT NULL, created_at TEXT NOT NULL);
                INSERT OR IGNORE INTO metadata VALUES ('generation', '0');
            """)
            if not db.execute("SELECT 1 FROM metadata WHERE key='initialized'").fetchone():
                if demo_corpus:
                    raw = json.dumps(demo_corpus, ensure_ascii=False).encode()
                    db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,1)",
                               ("demo", "مجموعة تعليمية تجريبية", "corpus.json", hashlib.sha256(raw).hexdigest(),
                                1, raw, self.now()))
                    db.executemany("INSERT INTO passages (id,document_id,ordinal,page,text,category) VALUES (?,'demo',?,1,?,?)",
                                   [(row["id"], n, row["text"], row["category"]) for n, row in enumerate(demo_corpus, 1)])
                db.execute("INSERT INTO metadata VALUES ('initialized', '1')")

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @property
    def generation(self) -> int:
        with self.connection() as db:
            return int(db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0])

    @staticmethod
    def _changed(db) -> None:
        db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='generation'")

    def documents(self) -> list[dict]:
        with self.connection() as db:
            return [dict(row) for row in db.execute("""
                SELECT d.id,d.title,d.filename,d.page_count,d.created_at,d.is_demo,
                       COUNT(p.id) AS passage_count FROM documents d LEFT JOIN passages p ON p.document_id=d.id
                GROUP BY d.id ORDER BY d.created_at DESC,d.id""")]

    def document(self, document_id: str, include_original: bool = False) -> dict | None:
        columns = "*" if include_original else "id,title,filename,page_count,created_at,is_demo"
        with self.connection() as db:
            row = db.execute(f"SELECT {columns} FROM documents WHERE id=?", (document_id,)).fetchone()
            return dict(row) if row else None

    def passages(self, document_id: str | None = None) -> list[dict]:
        where, args = ("WHERE p.document_id=?", (document_id,)) if document_id else ("", ())
        with self.connection() as db:
            return [dict(row) for row in db.execute(f"""
                SELECT p.*,d.title AS source_title,d.filename,d.is_demo FROM passages p
                JOIN documents d ON d.id=p.document_id {where} ORDER BY p.id""", args)]

    def passage(self, passage_id: int) -> dict | None:
        with self.connection() as db:
            row = db.execute("""SELECT p.*,d.title AS source_title,d.filename,d.is_demo FROM passages p
                JOIN documents d ON d.id=p.document_id WHERE p.id=?""", (passage_id,)).fetchone()
            return dict(row) if row else None

    def import_document(self, filename: str, content: bytes, replace_id: str | None = None) -> tuple[str, bool, str | None]:
        filename = re.split(r"[/\\]", filename)[-1].strip()[:160]
        digest = hashlib.sha256(content).hexdigest()
        pages = extract_pages(filename, content)
        chunks = chunk_pages(pages)
        if not chunks:
            raise DocumentError("لم يُعثر على مقاطع نصية قابلة للبحث.")
        document_id = replace_id or uuid.uuid4().hex
        with self.connection() as db:
            if replace_id and not db.execute("SELECT 1 FROM documents WHERE id=?", (replace_id,)).fetchone():
                raise DocumentError("المستند المطلوب تحديثه غير موجود.")
            existing = db.execute("SELECT id FROM documents WHERE sha256=?", (digest,)).fetchone()
            if existing:
                if replace_id and existing[0] != replace_id:
                    raise DocumentError("هذه النسخة موجودة بالفعل في مستند آخر؛ لم نغيّر المستند الأصلي.")
                return existing[0], False, None
            if replace_id:
                db.execute("DELETE FROM documents WHERE id=?", (replace_id,))
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                       (document_id, Path(filename).stem, filename, digest, len(pages), content, self.now()))
            db.executemany("INSERT INTO passages (document_id,ordinal,page,text) VALUES (?,?,?,?)",
                           [(document_id, row["ordinal"], row["page"], row["text"]) for row in chunks])
            self._changed(db)
        blank = sum(not text.strip() for _, text in pages)
        warning = f"لم يُستخرج نص من {blank} صفحة؛ راجع المصدر، فقد تحتاج هذه الصفحات OCR." if blank else None
        return document_id, True, warning

    def delete(self, document_id: str) -> bool:
        with self.connection() as db:
            deleted = db.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount > 0
            if deleted:
                self._changed(db)
            return deleted

    def add_feedback(self, passage_id: int, query: str, helpful: bool, method: str, dimension: int):
        with self.connection() as db:
            db.execute("INSERT INTO feedback (passage_id,query,helpful,method,dimension,created_at) VALUES (?,?,?,?,?,?)",
                       (passage_id, query, int(helpful), method, dimension, self.now()))
