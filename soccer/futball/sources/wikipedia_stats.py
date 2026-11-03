"""Season player statistics from Wikipedia.

No public-domain football API publishes per-league player statistics — FBref
lost its Opta feed in January 2026 and returns 403, and Understat bot-blocks
programmatic clients. Wikipedia's season articles, however, carry maintained
"Top scorers", "Clean sheets" and "Hat-tricks" tables, and the MediaWiki Action
API will render any section as HTML.

What is available and what is not, verified 2026-08-23:

    Premier League    Top scorers, Hat-tricks, Clean sheets
    La Liga           Top goalscorers, Hat-tricks
    Champions League  Top goalscorers

Assists are NOT published on these articles for any of the three, so the assist
board stays empty rather than being filled with a different statistic.

Rowspans matter here: a shared rank produces rows with fewer cells, so the
parser carries values down instead of mis-aligning the columns.
"""
from __future__ import annotations

import html as html_lib
import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from .. import config

log = logging.getLogger(__name__)

RAW_WIKI = config.RAW / "wikipedia"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "futball-dashboard/0.1 (https://github.com/futball-dashboard; personal analytics)"

# competition -> Wikipedia article title template
ARTICLES: dict[str, str] = {
    "E0": "{a}–{b} Premier League",
    "SP1": "{a}–{b} La Liga",
    "UCL": "{a}–{b} UEFA Champions League",
}

WANTED_SECTIONS = ("top scorers", "top goalscorers", "clean sheets", "hat-tricks")


@dataclass
class SeasonStats:
    competition_code: str
    season_year: int
    article: str
    scorers: list[dict]
    clean_sheets: list[dict]
    available: bool
    note: str | None = None


def _client() -> httpx.Client:
    return httpx.Client(timeout=40.0, follow_redirects=True,
                        headers={"User-Agent": USER_AGENT, "Accept": "application/json"})


def article_title(code: str, season_year: int) -> str:
    tpl = ARTICLES.get(code)
    if not tpl:
        raise ValueError(f"no Wikipedia article mapped for {code}")
    return tpl.format(a=season_year, b=str(season_year + 1)[2:])


def _sections(client: httpx.Client, page: str) -> list[dict]:
    r = client.get(API, params={"action": "parse", "format": "json",
                                "page": page, "prop": "sections"})
    if r.status_code != 200:
        return []
    return r.json().get("parse", {}).get("sections", []) or []


def _section_html(client: httpx.Client, page: str, index: str) -> str:
    r = client.get(API, params={"action": "parse", "format": "json",
                                "page": page, "section": index, "prop": "text"})
    if r.status_code != 200:
        return ""
    return r.json().get("parse", {}).get("text", {}).get("*", "")


def _clean(cell: str) -> str:
    cell = re.sub(r"<sup.*?</sup>", "", cell, flags=re.S)
    cell = re.sub(r"<style.*?</style>", "", cell, flags=re.S)
    cell = re.sub(r"<[^>]+>", "", cell)
    return html_lib.unescape(cell).strip()


def parse_wikitable(table_html: str) -> list[list[str]]:
    """Parse a wikitable, expanding rowspans so columns stay aligned."""
    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)
    grid: list[list[str]] = []
    carry: dict[int, tuple[str, int]] = {}  # column -> (value, rows remaining)

    for tr in rows_html:
        cells = re.findall(r"<(t[hd])([^>]*)>(.*?)</\1>", tr, re.S)
        row: list[str] = []
        col = 0
        # place any values still spanning from an earlier row
        while col in carry:
            val, left = carry[col]
            row.append(val)
            if left <= 1:
                del carry[col]
            else:
                carry[col] = (val, left - 1)
            col += 1
        for _tag, attrs, body in cells:
            text = _clean(body)
            m = re.search(r'rowspan\s*=\s*"?(\d+)', attrs)
            span = int(m.group(1)) if m else 1
            row.append(text)
            if span > 1:
                carry[col] = (text, span - 1)
            col += 1
            while col in carry:
                val, left = carry[col]
                row.append(val)
                if left <= 1:
                    del carry[col]
                else:
                    carry[col] = (val, left - 1)
                col += 1
        if any(c for c in row):
            grid.append(row)
    return grid


def _tables(section_html: str) -> list[list[list[str]]]:
    return [
        parse_wikitable(t)
        for t in re.findall(
            r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>.*?</table>', section_html, re.S
        )
    ]


def _pick(header: list[str], *names: str) -> int | None:
    low = [h.lower() for h in header]
    for n in names:
        for i, h in enumerate(low):
            if n in h:
                return i
    return None


def _rows_from(table: list[list[str]], value_names: tuple[str, ...]) -> list[dict]:
    if len(table) < 2:
        return []
    header = table[0]
    i_player = _pick(header, "player", "name")
    i_club = _pick(header, "club", "team")
    i_value = _pick(header, *value_names)
    if i_player is None or i_value is None:
        return []
    out: list[dict] = []
    for row in table[1:]:
        if max(i_player, i_value, i_club or 0) >= len(row):
            continue
        player = row[i_player].strip()
        raw = re.sub(r"[^\d]", "", row[i_value])
        if not player or not raw:
            continue
        out.append({
            "player": player,
            "team": row[i_club].strip() if i_club is not None and i_club < len(row) else None,
            "value": int(raw),
        })
    return out


def fetch(code: str, season_year: int, *, use_cache: bool = True) -> SeasonStats:
    page = article_title(code, season_year)
    dest = RAW_WIKI / f"{code}-{season_year}.json"
    if use_cache and dest.exists() and dest.stat().st_size > 0:
        try:
            d = json.loads(dest.read_text())
            return SeasonStats(**d)
        except Exception:  # noqa: BLE001
            pass

    scorers: list[dict] = []
    clean_sheets: list[dict] = []
    note = None

    with _client() as client:
        sections = _sections(client, page)
        if not sections:
            note = f"Wikipedia has no article titled “{page}” yet."
        for s in sections:
            line = (s.get("line") or "").lower()
            if not any(w in line for w in WANTED_SECTIONS):
                continue
            html_text = _section_html(client, page, s["index"])
            for table in _tables(html_text):
                if not scorers and ("scorer" in line):
                    scorers = _rows_from(table, ("goals",))
                if not clean_sheets and "clean sheet" in line:
                    clean_sheets = _rows_from(table, ("clean sheet", "clean"))
            time.sleep(0.3)

    stats = SeasonStats(
        competition_code=code, season_year=season_year, article=page,
        scorers=scorers, clean_sheets=clean_sheets,
        available=bool(scorers or clean_sheets),
        note=note or (None if scorers else f"No scorer table published on “{page}” yet."),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(stats.__dict__))
    return stats
