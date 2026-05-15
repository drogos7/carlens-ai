"""
Fetch approved public source URLs, extract OBD-style code mentions, and store
raw searchable text in the local SQLite knowledge DB.

Usage from backend/:
  .\\.venv\\Scripts\\python.exe knowledge\\ingest_sources.py knowledge\\sources.json

Create sources.json from sources.example.json and enable only URLs you are
allowed to crawl. This script does not bypass robots.txt or paywalls.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "knowledge" / "diagnostics.db"
USER_AGENT = "DiagnosticKnowledgeBot/0.1 (+local development)"
CODE_RE = re.compile(r"\b[PCBU][0-9A-Z]{4}\b", re.IGNORECASE)


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
    with urllib.request.urlopen(req, timeout=20) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Unsupported content type: {content_type}")
        return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def snippets_for_code(text: str, code: str, radius: int = 180) -> list[str]:
    snippets: list[str] = []
    for match in re.finditer(re.escape(code), text, flags=re.IGNORECASE):
        start = max(match.start() - radius, 0)
        end = min(match.end() + radius, len(text))
        snippets.append(" ".join(text[start:end].split()))
    return snippets[:5]


def store_document(url: str, title: str, text: str) -> tuple[int, int]:
    codes = sorted({code.upper().replace("O", "0") for code in CODE_RE.findall(text)})
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO source_documents (url, title, content, fetched_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(url) DO UPDATE SET
              title = excluded.title,
              content = excluded.content,
              fetched_at = CURRENT_TIMESTAMP
            """,
            (url, title, text),
        )
        doc_id = conn.execute("SELECT id FROM source_documents WHERE url = ?", (url,)).fetchone()[0]
        conn.execute("DELETE FROM code_mentions WHERE source_document_id = ?", (doc_id,))
        mention_count = 0
        for code in codes:
            for snippet in snippets_for_code(text, code):
                conn.execute(
                    "INSERT INTO code_mentions (code, source_document_id, snippet) VALUES (?, ?, ?)",
                    (code, doc_id, snippet),
                )
                mention_count += 1
        conn.commit()
    return len(codes), mention_count


def ingest(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for source in config.get("sources", []):
        if not source.get("enabled", False):
            continue
        url = source["url"]
        if not can_fetch(url):
            print(f"SKIP robots.txt: {url}")
            continue
        print(f"FETCH {url}")
        html = fetch_html(url)
        parser = TextExtractor()
        parser.feed(html)
        text = parser.text()
        code_count, mention_count = store_document(url, parser.title or source.get("name", ""), text)
        print(f"STORED {url}: {code_count} codes, {mention_count} mentions")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python knowledge/ingest_sources.py knowledge/sources.json")
    ingest(Path(sys.argv[1]))
