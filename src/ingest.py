"""Wikipedia ingestion using the public MediaWiki API directly (no wrapper libs).

We hit `action=query&prop=extracts&explaintext=1` to get clean plain-text
articles and persist each one as JSON under data/.

Robustness:
- Polite User-Agent with contact info (Wikipedia requires this).
- Exponential backoff on HTTP 429 / 5xx.
- Skips titles that are already on disk so re-runs resume rather than restart.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable

import requests

from config import DATA_DIR

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "LocalWikipediaRAG/1.0 (ITU BLG483E homework; "
    "Muhammet Ali Yagiz, student id 820220327)"
)


def _slug(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return s or "untitled"


def fetch_article(title: str, max_retries: int = 5) -> dict | None:
    """Fetch a single Wikipedia article as plain text, with backoff on 429/5xx."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "formatversion": 2,
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            print(f"    network error ({e}); retry {attempt}/{max_retries} in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
            continue

        if r.status_code == 200:
            payload = r.json()
            pages = payload.get("query", {}).get("pages", [])
            if not pages:
                return None
            page = pages[0]
            if page.get("missing"):
                return None
            text = page.get("extract", "")
            if not text.strip():
                return None
            return {
                "title": page.get("title", title),
                "pageid": page.get("pageid"),
                "text": text,
                "url": f"https://en.wikipedia.org/wiki/{page.get('title', title).replace(' ', '_')}",
            }

        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get("Retry-After", backoff))
            print(f"    HTTP {r.status_code}; backing off {wait:.0f}s "
                  f"(attempt {attempt}/{max_retries})")
            time.sleep(wait)
            backoff *= 2
            continue

        # Non-retryable
        r.raise_for_status()
        return None

    print(f"    gave up on {title} after {max_retries} retries")
    return None


def ingest_entities(entities: Iterable[str], entity_type: str,
                    out_dir: Path = DATA_DIR, force: bool = False) -> list[Path]:
    """Download articles for an iterable of titles and persist them as JSON.

    Skips titles that already have a JSON file unless `force=True`.
    """
    out_dir = Path(out_dir) / entity_type
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for title in entities:
        path = out_dir / f"{_slug(title)}.json"
        if path.exists() and not force:
            print(f"  = {entity_type:6s}  {title}  [cached, skipping]")
            saved.append(path)
            continue
        try:
            article = fetch_article(title)
        except Exception as e:
            print(f"  ! error fetching {title}: {e}")
            continue
        if not article:
            print(f"  ! no article for {title}")
            continue
        article["type"] = entity_type
        path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(path)
        print(f"  + {entity_type:6s}  {article['title']}  ({len(article['text']):,} chars)")
        time.sleep(1.2)  # be polite — well below Wikipedia's per-IP limits
    return saved


def load_all(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load every ingested article from disk."""
    docs = []
    for sub in ("person", "place"):
        folder = Path(data_dir) / sub
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.json")):
            docs.append(json.loads(f.read_text(encoding="utf-8")))
    return docs
