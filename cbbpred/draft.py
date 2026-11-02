"""2027 NBA draft inputs: consensus big board, who owns each pick, lottery odds, and NBA team strength.

ESPN's own 2027 draft is empty this early, so the board, pick ownership and lottery weights are read from
Tankathon's public pages. The simulation itself (season, lottery, draft) runs in the browser from these inputs.
A page that no longer parses raises ``DraftFeedError`` so the caller keeps the last good file.
"""

from __future__ import annotations

import html
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from cbbpred.store import read_json

log = logging.getLogger(__name__)

SITE = "https://www.tankathon.com"
PAGES = {"board": "/big-board", "order": "/full-draft", "odds": "/pick-odds"}
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15"
MIN_BOARD = 30
PICKS_PER_ROUND = 30
LOTTERY_TEAMS = 16
NAME_SUFFIX = {"jr", "sr", "ii", "iii", "iv"}


class DraftFeedError(RuntimeError):
    """A source page is missing or its markup changed, so the draft inputs cannot be trusted."""


def fetch_html(path: str) -> str:
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = httpx.get(SITE + path, headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)
            if resp.status_code == 200:
                return resp.text
            last = DraftFeedError(f"HTTP {resp.status_code} for {path}")
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(1.5 * 2**attempt)
    raise DraftFeedError(str(last))


def height_in(text: str) -> float | None:
    m = re.match(r"\s*(\d+)'\s*([\d.]+)?", text)
    return int(m.group(1)) * 12 + float(m.group(2) or 0) if m else None


def norm_name(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z ]", " ", plain).split()
    return " ".join(w for w in words if w not in NAME_SUFFIX)


def parse_board(page: str) -> list[dict[str, Any]]:
    """Prospects in consensus order. International players carry a country in ``intl`` and have no college stats."""
    rows = []
    for block in re.split(r'<div class="mock-row"', page)[1:]:
        rank = re.search(r'mock-row-pick-number">(\d+)<', block)
        slug = re.search(r'href="/players/([^"]+)"', block)
        name = re.search(r'mock-row-name">([^<]+)<', block)
        sp = re.search(r'mock-row-school-position">([^|<]+)\|([^<]*)<', block)
        if not (rank and slug and name and sp):
            continue
        hw = re.search(r'height-weight"><div>([^<]+)</div><div>(\d+)', block)
        ya = re.search(r'year-age[^"]*"><div>([^<]+)</div><div>([\d.]+) yrs', block)
        flag = re.search(r'flag-border" alt="([^"]+)"', block)
        rows.append({
            "rank": int(rank.group(1)), "slug": slug.group(1), "name": html.unescape(name.group(1)).strip(),
            "pos": sp.group(1).strip(), "school": html.unescape(sp.group(2)).strip(),
            "height_in": height_in(html.unescape(hw.group(1))) if hw else None, "weight": int(hw.group(2)) if hw else None,
            "cls": ya.group(1) if ya else None, "age": float(ya.group(2)) if ya else None,
            "intl": html.unescape(flag.group(1)) if flag else None,
        })
    unique = list({r["slug"]: r for r in reversed(rows)}.values())      # the page repeats the list; first copy wins
    if len(unique) < MIN_BOARD:
        raise DraftFeedError(f"big board parsed {len(unique)} prospects, expected at least {MIN_BOARD}")
    return sorted(unique, key=lambda r: r["rank"])


def parse_order(page: str) -> dict[str, dict[str, str]]:
    """origin team slug -> {"r1": owner slug, "r2": owner slug}. A pick marked 'from' is owned by someone else."""
    owners: dict[str, dict[str, str]] = {}
    rounds = re.split(r'<div class="round-title">', page)[1:]
    for key, chunk in zip(("r1", "r2"), rounds):
        for row in re.findall(r'<tr><td class="pick-number">.*?</tr>', chunk, flags=re.S):
            owner = re.search(r'<div class="team-link"><a href="/([^"]+)"', row)
            origin = re.search(r'<div class="trade">.*?href="/([^"]+)"', row, flags=re.S)
            if owner:
                owners.setdefault((origin or owner).group(1), {})[key] = owner.group(1)
    for key in ("r1", "r2"):
        if sum(key in v for v in owners.values()) != PICKS_PER_ROUND:
            raise DraftFeedError(f"draft order has the wrong number of {key} picks")
    return owners


def parse_odds(page: str) -> dict[str, Any]:
    """Lottery balls per seed (worst record first) and teams barred from the top picks.

    The table shows each seed's chance at pick 1, renormalised without barred teams, so the ratio between seeds is
    the ratio of balls. A barred seed shows no pick-1 chance; it takes the next seed's ball count (worse record
    never gets fewer balls).
    """
    chance: list[float] = []
    restricted: dict[str, int] = {}
    for row in re.findall(r'<tr><td class="team">.*?</tr>', page, flags=re.S):
        slug = re.search(r'<a href="/([^"]+)"', row)
        cells = re.findall(r'<td class="percent([^"]*)"[^>]*>(.*?)</td>', row, flags=re.S)[:LOTTERY_TEAMS]
        barred = sum(1 for cls, _ in cells if "ineligible" in cls)
        first = re.sub(r"<[^>]+>", "", cells[0][1]).strip() if cells else ""
        chance.append(float(first) if re.fullmatch(r"\d+(\.\d+)?", first) else 0.0)
        if barred and slug:
            restricted[slug.group(1)] = barred
    if len(chance) != LOTTERY_TEAMS:
        raise DraftFeedError(f"lottery table parsed {len(chance)} seeds, expected {LOTTERY_TEAMS}")
    unit = min(c for c in chance if c > 0)
    balls = [round(c / unit) for c in chance]
    for i in range(LOTTERY_TEAMS - 1, -1, -1):
        balls[i] = balls[i] or (balls[i + 1] if i + 1 < LOTTERY_TEAMS else 1)
    return {"balls": balls, "restricted": restricted}


def nba_teams(standings_path: Path) -> list[dict[str, Any]]:
    """NBA team strength from the app's own standings; ``slug`` matches Tankathon's team links."""
    rows = read_json(standings_path)["power_rankings"]
    return [{
        "slug": r["nickname"].lower().replace(" ", "-"), "abbr": r["abbrev"], "name": r["name"], "logo": r["logo"],
        "w": r["wins"], "l": r["losses"], "power": r["power"],
    } for r in rows]


def link_espn(board: list[dict[str, Any]], players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach ESPN's athlete id so the board can show headshots and stats. Ambiguous names must match the school."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for p in players:
        by_name.setdefault(norm_name(p["name"] or ""), []).append(p)
    out = []
    for b in board:
        found = [] if b["intl"] else by_name.get(norm_name(b["name"]), [])
        school = norm_name(b["school"])
        hit = next((p for p in found if school and school in norm_name(p.get("team") or "")), None)
        hit = hit or (found[0] if len(found) == 1 else None)
        out.append({**b, "espn_id": hit["id"] if hit else None})
    return out


def draft_payload(players: list[dict[str, Any]], standings_path: Path, generated_at: str, year: int,
                  fetch: Callable[[str], str] = fetch_html) -> dict[str, Any]:
    board = parse_board(fetch(PAGES["board"]))
    owners = parse_order(fetch(PAGES["order"]))
    lottery = parse_odds(fetch(PAGES["odds"]))
    owners_known = set(owners) | set(lottery["restricted"])
    teams = nba_teams(standings_path)
    unknown = owners_known | {o for v in owners.values() for o in v.values()}
    unknown -= {t["slug"] for t in teams}
    if unknown:
        raise DraftFeedError(f"unknown NBA team slugs: {sorted(unknown)}")
    return {"generated_at": generated_at, "draft_year": year, "board": link_espn(board, players),
            "teams": teams, "owners": owners, "lottery": lottery,
            "source": "Tankathon consensus board and pick ownership; NBA strength from this app's standings"}
