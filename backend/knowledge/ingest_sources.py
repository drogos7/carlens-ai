"""
Fetch approved public source URLs, extract OBD codes and VINs, store searchable
text in the local SQLite knowledge DB.

Usage from backend/:
  .\\.venv\\Scripts\\python.exe knowledge\\ingest_sources.py knowledge\\sources.mercedes.json

Respects robots.txt. Does not bypass paywalls or login walls.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
BRANDS_DIR = KNOWLEDGE_DIR / "brands"
DB_PATH = KNOWLEDGE_DIR / "diagnostics.db"
USER_AGENT = "CarLensAI-KnowledgeBot/1.0 (+local research; contact project owner)"
CODE_RE = re.compile(r"\b[PCBU][0-3][0-9A-F]{3}\b", re.IGNORECASE)
VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)
LINK_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._skip_depth:
            return
        if self._in_title:
            self.title = (self.title + " " + text).strip()
        self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            brand_slug TEXT NOT NULL DEFAULT '',
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    doc_cols = {row[1] for row in conn.execute("PRAGMA table_info(source_documents)").fetchall()}
    if doc_cols and "brand_slug" not in doc_cols:
        conn.execute(
            "ALTER TABLE source_documents ADD COLUMN brand_slug TEXT NOT NULL DEFAULT ''"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS code_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            brand_slug TEXT NOT NULL DEFAULT '',
            source_document_id INTEGER NOT NULL,
            snippet TEXT NOT NULL,
            FOREIGN KEY (source_document_id) REFERENCES source_documents(id)
        )
        """
    )
    mention_cols = {row[1] for row in conn.execute("PRAGMA table_info(code_mentions)").fetchall()}
    if mention_cols and "brand_slug" not in mention_cols:
        conn.execute(
            "ALTER TABLE code_mentions ADD COLUMN brand_slug TEXT NOT NULL DEFAULT ''"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vin_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vin TEXT NOT NULL,
            source_document_id INTEGER NOT NULL,
            snippet TEXT NOT NULL,
            FOREIGN KEY (source_document_id) REFERENCES source_documents(id)
        )
        """
    )


def can_fetch(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    robots_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception:
        return False
    return parser.can_fetch(USER_AGENT, url)


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=25) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Unsupported content type: {content_type}")
        return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def normalize_code(code: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", code).upper()
    if len(cleaned) >= 2:
        cleaned = cleaned[0] + cleaned[1:].replace("O", "0").replace("I", "1")
    return cleaned


def normalize_vin(vin: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", vin).upper().replace("O", "0").replace("I", "1").replace("Q", "0")[:17]


def snippets_for_term(text: str, term: str, radius: int = 200) -> list[str]:
    snippets: list[str] = []
    for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
        start = max(match.start() - radius, 0)
        end = min(match.end() + radius, len(text))
        snippet = " ".join(text[start:end].split())
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= 5:
            break
    return snippets


def extract_links(html: str, base_url: str, keywords: list[str]) -> list[str]:
    parsed_base = urllib.parse.urlparse(base_url)
    found: list[str] = []
    for href in LINK_RE.findall(html):
        joined = urllib.parse.urljoin(base_url, href)
        link = urllib.parse.urljoin(joined, urllib.parse.urlparse(joined).path)
        link_parsed = urllib.parse.urlparse(link)
        if link_parsed.netloc != parsed_base.netloc:
            continue
        if link_parsed.scheme not in {"http", "https"}:
            continue
        lower = link.lower()
        if any(keyword in lower for keyword in keywords):
            if link not in found:
                found.append(link)
    return found


def store_document(url: str, title: str, text: str, brand_slug: str = "") -> tuple[int, int, int]:
    codes = sorted({normalize_code(code) for code in CODE_RE.findall(text) if len(normalize_code(code)) == 5})
    vins = sorted({normalize_vin(vin) for vin in VIN_RE.findall(text) if len(normalize_vin(vin)) == 17})

    with sqlite3.connect(DB_PATH) as conn:
        ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO source_documents (url, title, content, brand_slug, fetched_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(url) DO UPDATE SET
              title = excluded.title,
              content = excluded.content,
              brand_slug = excluded.brand_slug,
              fetched_at = CURRENT_TIMESTAMP
            """,
            (url, title, text, brand_slug),
        )
        doc_id = conn.execute("SELECT id FROM source_documents WHERE url = ?", (url,)).fetchone()[0]
        conn.execute("DELETE FROM code_mentions WHERE source_document_id = ?", (doc_id,))
        conn.execute("DELETE FROM vin_mentions WHERE source_document_id = ?", (doc_id,))

        code_mentions = 0
        for code in codes:
            for snippet in snippets_for_term(text, code):
                conn.execute(
                    "INSERT INTO code_mentions (code, brand_slug, source_document_id, snippet) VALUES (?, ?, ?, ?)",
                    (code, brand_slug, doc_id, snippet),
                )
                code_mentions += 1

        vin_mentions = 0
        for vin in vins:
            for snippet in snippets_for_term(text, vin):
                conn.execute(
                    "INSERT INTO vin_mentions (vin, source_document_id, snippet) VALUES (?, ?, ?)",
                    (vin, doc_id, snippet),
                )
                vin_mentions += 1

        conn.commit()
    return len(codes), code_mentions, vin_mentions


def resolve_config_path(arg: str) -> tuple[Path, str]:
    """Accept brand slug (e.g. mercedes-benz) or a path to sources.json."""
    path = Path(arg)
    if path.is_file():
        slug = path.parent.name if path.parent.parent.name == "brands" else ""
        return path, slug
    brand_sources = BRANDS_DIR / arg / "sources.json"
    if brand_sources.is_file():
        return brand_sources, arg
    legacy = KNOWLEDGE_DIR / f"sources.{arg}.json"
    if legacy.is_file():
        return legacy, arg.replace("mercedes", "mercedes-benz")
    raise FileNotFoundError(
        f"No sources config for {arg!r}. Expected {brand_sources} or a file path."
    )


def ingest(config_path: Path, brand_slug: str = "") -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config.get("settings", {})
    delay = float(settings.get("request_delay_seconds", 2.0))
    max_per_source = int(settings.get("max_pages_per_source", 25))
    max_total = int(settings.get("max_total_pages", 120))
    keywords = [k.lower() for k in settings.get("link_keywords", [])]
    follow_default = bool(settings.get("follow_same_domain_links", True))

    visited: set[str] = set()
    total_pages = 0

    for source in config.get("sources", []):
        if not source.get("enabled", False):
            print(f"SKIP disabled: {source.get('name', source.get('url'))}")
            continue

        seed_url = source["url"]
        queue = [seed_url]
        source_pages = 0

        while queue and source_pages < max_per_source and total_pages < max_total:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            if not can_fetch(url):
                print(f"SKIP robots.txt: {url}")
                continue

            print(f"FETCH {url}")
            try:
                html = fetch_html(url)
            except Exception as exc:
                print(f"ERROR {url}: {exc}")
                continue

            parser = TextExtractor()
            parser.feed(html)
            text = parser.text()
            code_count, code_mentions, vin_mentions = store_document(
                url, parser.title or source.get("name", ""), text, brand_slug=brand_slug
            )
            source_pages += 1
            total_pages += 1
            print(f"STORED {url}: {code_count} codes ({code_mentions} mentions), {vin_mentions} vin mentions")

            if source.get("follow_links", follow_default) and keywords:
                for link in extract_links(html, url, keywords):
                    if link not in visited and link not in queue:
                        queue.append(link)

            time.sleep(delay)

    print(f"Done. Pages fetched: {total_pages}, unique URLs visited: {len(visited)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python knowledge/ingest_sources.py <brand-slug|sources.json>\n"
            "  e.g. python knowledge/ingest_sources.py mercedes-benz"
        )
    config_path, brand_slug = resolve_config_path(sys.argv[1])
    print(f"Ingesting for brand: {brand_slug or '(unspecified)'} from {config_path}")
    ingest(config_path, brand_slug=brand_slug)
