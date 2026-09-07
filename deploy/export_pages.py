"""Export the bundled public demo as a static GitHub Pages site."""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "static"
ATTRIBUTE_URL = re.compile(
    r"(?P<prefix>\b(?:href|src|action)\s*=\s*)(?P<quote>['\"])(?P<url>/[^'\"]*)(?P=quote)",
    re.IGNORECASE,
)


def normalize_base_path(value: str) -> str:
    """Return a safe absolute Pages prefix with exactly one trailing slash."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base path must not be empty")
    candidate = value.strip()
    if any(character in candidate for character in ("?", "#", "\\")) or candidate.startswith("//"):
        raise ValueError("base path must be a URL path")
    parts = [part for part in candidate.split("/") if part]
    if (not parts or any(unquote(part) in (".", "..") for part in parts)
            or any(not re.fullmatch(r"(?:[A-Za-z0-9._~-]|%[0-9A-Fa-f]{2})+", part) for part in parts)):
        raise ValueError("base path must name a site directory")
    return "/" + "/".join(parts) + "/"


def validate_api_base(value: str) -> str:
    """Accept an empty API setting or a path-free HTTPS origin."""
    if not isinstance(value, str):
        raise ValueError("API base must be a string")
    candidate = value.strip()
    if not candidate:
        return ""
    if (any(character.isspace() or ord(character) < 32 for character in candidate)
            or "*" in candidate or "\\" in candidate):
        raise ValueError("API base must not contain whitespace")
    parsed = urlsplit(candidate)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment):
        raise ValueError("API base must be an HTTPS origin without a path, query, or credentials")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("API base has an invalid port") from exc
    return candidate.rstrip("/")


def _prepare_output(output: Path) -> Path:
    if output.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise FileExistsError(f"output directory must be empty: {output}")
    else:
        output.mkdir(parents=True)
    return output.resolve()


def _copy_public_assets(output: Path) -> None:
    pages_script = STATIC_ROOT / "js" / "pages.js"
    if not pages_script.is_file():
        raise FileNotFoundError(f"static Pages client is missing: {pages_script}")
    for source in STATIC_ROOT.rglob("*"):
        if source.is_symlink():
            raise ValueError(f"static assets must not contain symbolic links: {source}")
        if source.is_file():
            relative = source.relative_to(STATIC_ROOT)
            destination = output / "static" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    (output / ".nojekyll").write_text("", encoding="utf-8")


def _root_url(url: str, base_path: str, page_paths: set[str]) -> str:
    if url.startswith("//"):
        return url
    positions = [position for marker in ("?", "#") if (position := url.find(marker)) >= 0]
    boundary = min(positions) if positions else len(url)
    path, suffix = url[:boundary], url[boundary:]
    route = path.rstrip("/") or "/"
    if route == "/":
        return base_path + suffix
    target = base_path + path.lstrip("/").rstrip("/")
    if route in page_paths:
        target += "/"
    return target + suffix


def _rewrite_html(source: str, base_path: str, api_base: str, page_paths: set[str]) -> str:
    # The static client owns search behavior; loading both clients would attach two handlers.
    source = re.sub(
        r"(?P<prefix>\bsrc\s*=\s*)(?P<quote>['\"])/static/js/app\.js(?P=quote)",
        r"\g<prefix>\g<quote>/static/js/pages.js\g<quote>",
        source,
        flags=re.IGNORECASE,
    )

    def replace_attribute(match: re.Match[str]) -> str:
        rewritten = _root_url(match.group("url"), base_path, page_paths)
        return f'{match.group("prefix")}{match.group("quote")}{rewritten}{match.group("quote")}'

    source = ATTRIBUTE_URL.sub(replace_attribute, source)
    metadata = (
        f'  <meta name="bahith-base-path" content="{html.escape(base_path, quote=True)}">\n'
        f'  <meta name="bahith-api-base" content="{html.escape(api_base, quote=True)}">\n'
    )
    if "</head>" not in source:
        raise ValueError("rendered page has no head element")
    return source.replace("</head>", metadata + "</head>", 1)


def _replace_main(source: str, content: str, title: str) -> str:
    replacement = f'<main id="main-content"><div class="container">{content}</div></main>'
    source, count = re.subn(
        r'<main id="main-content">.*?</main>',
        lambda _match: replacement,
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("rendered page has no main content element")
    return re.sub(r"<title>.*?</title>", f"<title>{html.escape(title)}</title>", source, count=1,
                  flags=re.DOTALL)


def _write_route(output: Path, route: str, content: str) -> Path:
    relative = route.strip("/")
    destination = output / relative / "index.html" if relative else output / "index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def _load_create_app():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app import create_app
    return create_app


def export_pages(output_dir: str | Path, *, base_path: str = "/bahith/",
                 api_base: str = "") -> list[Path]:
    """Render the public demo into a new or empty directory and return written HTML files."""
    base_path = normalize_base_path(base_path)
    api_base = validate_api_base(api_base)
    output = _prepare_output(Path(output_dir).expanduser())
    rendered: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="bahith-pages-config-") as configured_directory:
        safe_environment = {
            "BAHITH_PUBLIC_DEMO": "1",
            "BAHITH_PRELOAD_MODEL": "0",
            "BAHITH_ALLOWED_HOSTS": "testserver",
            "BAHITH_ALLOWED_ORIGINS": "",
            "BAHITH_ALLOW_HF_EMBED": "0",
        }
        with patch.dict(os.environ, safe_environment, clear=False):
            from fastapi.testclient import TestClient

            create_app = _load_create_app()
            application = create_app(configured_directory, model=None, seed_demo=True, public_demo=True)
            with TestClient(application) as client:
                passages = application.state.store.passages()
                passage_ids = [int(row["id"]) for row in passages]
                categories = sorted({str(row["category"]) for row in passages})
                routes = ["/", "/search", "/documents", "/documents/demo", "/browse", "/about"]
                routes.extend(f"/browse/{quote(category, safe='')}" for category in categories)
                routes.extend(f"/passages/{passage_id}" for passage_id in passage_ids)

                for route in routes:
                    response = client.get(route)
                    if response.status_code != 200:
                        raise RuntimeError(f"could not render {route}: HTTP {response.status_code}")
                    rendered[route] = response.text

                source_response = client.get("/documents/demo/source")
                if source_response.status_code != 200:
                    raise RuntimeError(f"could not render demo source: HTTP {source_response.status_code}")
                source_content = (
                    '<nav class="breadcrumb" aria-label="مسار الصفحة"><a href="/documents/demo">'
                    'المجموعة التعليمية التجريبية</a><span aria-hidden="true">←</span><span>المصدر</span></nav>'
                    '<header class="page-head page-head-compact"><p class="eyebrow">المصدر المضمّن</p>'
                    '<h1>corpus.json</h1><p>النص الأصلي للمجموعة التعليمية التجريبية.</p></header>'
                    '<section class="prose"><pre dir="ltr" lang="ar" style="white-space:pre-wrap;overflow-wrap:anywhere">'
                    f'{html.escape(source_response.text)}</pre></section>'
                )
                rendered["/documents/demo/source"] = _replace_main(
                    rendered["/documents/demo"], source_content, "المصدر المضمّن · باحث"
                )

    page_paths = set(rendered)
    written = []
    no_script = (
        '<noscript><div class="state-card state-error" role="alert"><h2>يلزم JavaScript للبحث المباشر</h2>'
        '<p>يبقى تصفح الوثيقة والمقاطع متاحًا من روابط الموقع.</p></div></noscript>'
    )
    for route, page in rendered.items():
        if route == "/search":
            page = page.replace('<section id="results-region"', no_script + '<section id="results-region"', 1)
        page = _rewrite_html(page, base_path, api_base, page_paths)
        written.append(_write_route(output, route, page))

    missing_content = (
        '<header class="page-head"><p class="eyebrow">404</p><h1>الصفحة غير موجودة</h1>'
        '<p>قد يكون الرابط قديمًا. يمكنك العودة إلى الصفحة الرئيسية أو تصفح النصوص المضمّنة.</p></header>'
        '<div class="section-actions"><a class="button button-primary" href="/">الصفحة الرئيسية</a>'
        '<a class="button button-secondary" href="/browse">تصفح النصوص</a></div>'
    )
    missing_page = _replace_main(rendered["/"], missing_content, "الصفحة غير موجودة · باحث")
    missing_page = _rewrite_html(missing_page, base_path, api_base, page_paths)
    missing_path = output / "404.html"
    missing_path.write_text(missing_page, encoding="utf-8")
    written.append(missing_path)

    _copy_public_assets(output)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new or empty output directory")
    parser.add_argument("--base-path", default="/bahith/", help="GitHub Pages project path")
    parser.add_argument("--api-base", default="", help="optional path-free HTTPS API origin")
    arguments = parser.parse_args(argv)
    export_pages(arguments.output, base_path=arguments.base_path, api_base=arguments.api_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
