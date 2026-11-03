"""API-Football player portraits, cached on the T7.

Only provider portraits are served. Missing players use the frontend's initials
placeholder; Wikipedia and unidentified legacy images are not displayed.
Squad lookups use API quota; image downloads use the provider's media CDN.
Images retain their providers' licences.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

from functools import lru_cache

from . import config
from .storage import data_files, write_text_atomic
from .teams import normalise

log = logging.getLogger(__name__)

CACHE_DIR = config.CACHE / "headshots"
MANIFEST = CACHE_DIR / "manifest.json"
USER_AGENT = "futball-dashboard/0.1 (player portraits)"

# Refresh a miss occasionally; a hit essentially never changes.
MISS_TTL_SECONDS = 14 * 24 * 3600

_lock = threading.Lock()
_manifest: dict[str, dict] | None = None


@dataclass
class Headshot:
    name: str
    found: bool
    image_url: str | None = None
    article: str | None = None
    page_url: str | None = None
    file: str | None = None
    checked_at: float = 0.0
    # Which provider supplied the image. Recorded so a wrong face can be traced
    # to its source rather than being an anonymous mystery in the cache.
    provider: str | None = None


def _load() -> dict[str, dict]:
    global _manifest
    if _manifest is None:
        if MANIFEST.exists():
            try:
                _manifest = json.loads(MANIFEST.read_text())
            except Exception:  # noqa: BLE001
                _manifest = {}
        else:
            _manifest = {}
    return _manifest


def _save() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_text_atomic(MANIFEST, json.dumps(_load(), indent=1))


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def _slug(name: str) -> str:
    return hashlib.sha256(_key(name).encode()).hexdigest()[:20]


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=25.0,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
    )


def _download(client: httpx.Client, url: str, name: str) -> str | None:
    try:
        r = client.get(url)
        if r.status_code != 200 or not r.content:
            return None
    except Exception:  # noqa: BLE001
        return None
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        ext = ".jpg"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{_slug(name)}{ext}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return dest.name


# ---------------------------------------------------------------------------
# provider 1: API-Football squad portraits
# ---------------------------------------------------------------------------

def _initial_key(name: str) -> str | None:
    """"Bukayo Saka" -> "b saka", matching the squad feed's own abbreviation.

    API-Football abbreviates the given name for 68% of squad entries ("B. Saka",
    "M. Ødegaard"), so an index keyed on full names reaches only a third of
    players. Reducing both sides to initial-plus-surname bridges them.
    """
    tokens = normalise(name or "").split()
    if len(tokens) < 2 or not tokens[0]:
        return None
    return f"{tokens[0][0]} {' '.join(tokens[1:])}"


def _build(rows: list[dict], key_of) -> dict[str, str]:
    """Index squad rows by `key_of`, dropping keys that two players share.

    Ambiguity is resolved by *removal*, never by taking whichever club was read
    first: a portrait is a claim about who someone is, and the wrong face is
    considerably worse than no face.
    """
    index: dict[str, str] = {}
    clashes: set[str] = set()
    for row in rows:
        photo, name = row.get("photo"), row.get("name")
        if not photo or not name:
            continue
        key = key_of(name)
        if not key:
            continue
        if key in index and index[key] != photo:
            clashes.add(key)
            continue
        index.setdefault(key, photo)
    for key in clashes:
        index.pop(key, None)
    return index


@lru_cache(maxsize=1)
def _portrait_index() -> dict[str, str]:
    """Normalised full player name -> portrait URL."""
    from .sources import api_football as af
    return _build(af.cached_squads(), normalise)


@lru_cache(maxsize=1)
def _initial_index() -> dict[str, str]:
    """Initial-plus-surname -> portrait URL, for the abbreviated majority."""
    from .sources import api_football as af
    return _build(af.cached_squads(), _initial_key)


def portrait_url(name: str) -> str | None:
    """API-Football portrait for a player, or None.

    StatsBomb records full legal names ("Lamine Yamal Nasraoui Ebana") where
    squad lists carry the common form ("Lamine Yamal"), so an exact miss falls
    back to a squad name that is a leading run of the given one. At least two
    tokens must match, or short surnames would attach the wrong face.
    """
    key = normalise(name or "")
    if not key:
        return None

    index = _portrait_index()
    hit = index.get(key)
    if hit:
        return hit

    # A leading run of the full name: "Lamine Yamal Nasraoui Ebana" -> "Lamine Yamal".
    tokens = key.split()
    for size in range(len(tokens) - 1, 1, -1):
        hit = index.get(" ".join(tokens[:size]))
        if hit:
            return hit

    # Initial plus surname, longest form first so the most specific wins.
    initials = _initial_index()
    if len(tokens) >= 2:
        first = tokens[0][0]
        for size in range(len(tokens), 1, -1):
            hit = initials.get(f"{first} {' '.join(tokens[1:size])}")
            if hit:
                return hit
    return None


def refresh_caches() -> None:
    """Drop the memoised portrait indexes after the raw layer is rewritten."""
    _portrait_index.cache_clear()
    _initial_index.cache_clear()


def resolve(name: str, *, download: bool = True) -> Headshot:
    """Use provider portraits only, replacing any legacy Wikipedia cache hit."""
    key = _key(name)
    if not key:
        return Headshot(name=name, found=False)
    image_url = portrait_url(name)
    with _lock:
        cached = _load().get(key)
    if cached and cached.get("provider") == "api-football":
        h = Headshot(**cached)
        has_file = bool(h.file) and (CACHE_DIR / h.file).is_file()
        same_portrait = image_url is None or image_url == h.image_url
        if h.found and same_portrait and (has_file or not download):
            return h
        # Retry temporary CDN failures after five minutes; a newly discovered
        # portrait bypasses a cached miss immediately.
        retry_after = 300 if h.image_url else MISS_TTL_SECONDS
        if not h.found and same_portrait and time.time() - h.checked_at < retry_after:
            return h
        image_url = image_url or h.image_url

    file_name = None
    if image_url and download:
        with _client() as client:
            file_name = _download(client, image_url, name)
    h = Headshot(
        name=name, found=bool(file_name) if download else bool(image_url),
        image_url=image_url, article=name, file=file_name,
        checked_at=time.time(), provider="api-football",
    )
    with _lock:
        _load()[key] = asdict(h)
        _save()
    return h


def resolve_many(names: list[str], *, download: bool = True) -> dict[str, Headshot]:
    out: dict[str, Headshot] = {}
    for n in names:
        if not n:
            continue
        out[n] = resolve(n, download=download)
    return out


def cached_path(name: str) -> Path | None:
    with _lock:
        rec = _load().get(_key(name))
    if not rec or rec.get("provider") != "api-football" or not rec.get("found") or not rec.get("file"):
        return None
    p = CACHE_DIR / rec["file"]
    return p if p.exists() else None


def stats() -> dict:
    m = _load()
    found = sum(1 for v in m.values() if v.get("found") and v.get("provider") == "api-football")
    files = len(list(data_files(CACHE_DIR, "*"))) - (1 if MANIFEST.exists() else 0)
    return {"names": len(m), "found": found, "missing": len(m) - found,
            "files_cached": max(files, 0)}


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

def backfill_squads(limit: int | None = None, *, reserve: int = 5) -> dict[str, int]:
    """Cache squad rosters so portraits can be resolved offline.

    Sized against the live request budget rather than assuming a full 100/day:
    the free plan is shared with anything else using the key, and walking into
    the cap halfway through leaves a partial index. Already-cached clubs cost
    nothing, so re-running this simply continues where it stopped.

    `reserve` keeps a few requests back so a later interactive lookup is not
    blocked by a backfill that used the very last one.
    """
    from . import clubs
    from .sources import api_football as af

    report = {"clubs": 0, "fetched": 0, "cached": 0, "players": 0, "portraits": 0,
              "searched": 0, "unmapped": 0, "skipped_budget": 0, "failed": 0}
    if not af.available():
        log.info("api-football: no API_FOOTBALL_KEY, skipping portrait backfill")
        return report

    budget = af.requests_remaining()
    if budget is not None:
        budget = max(0, budget - reserve)

    # Team ids first: three cheap calls that make every squad lookup possible.
    team_ids: dict[str, int] = {}
    for code in ("EPL", "SP1", "UCL"):
        try:
            team_ids.update(af.fetch_team_ids(code))
        except Exception as exc:  # noqa: BLE001 - one competition must not stop the rest
            log.warning("api-football team ids %s failed: %s", code, exc)

    resolver_names = list(clubs.directory())
    report["clubs"] = len(resolver_names)

    for club_name in resolver_names:
        entry = clubs.directory()[club_name]
        team_id = _match_team_id(club_name, entry, team_ids)
        if team_id is None:
            # Not in the enumerated leagues: a promoted side, or a Champions
            # League entrant from a competition this project does not ingest.
            # One search request each reaches them; skip if the budget is gone.
            if budget is not None and budget <= 1:
                report["unmapped"] += 1
                continue
            team_id = af.search_team_id(entry.get("short_name") or club_name)
            if budget is not None:
                budget -= 1
            if team_id is None:
                report["unmapped"] += 1
                continue
            report["searched"] = report.get("searched", 0) + 1

        cached = af.squad_is_cached(team_id)
        if not cached:
            if budget is not None and budget <= 0:
                report["skipped_budget"] += 1
                continue
            if limit is not None and report["fetched"] >= limit:
                report["skipped_budget"] += 1
                continue

        try:
            rows = af.fetch_squad(team_id)
        except Exception as exc:  # noqa: BLE001 - one club must not stop the rest
            log.warning("api-football squad %s (%s) failed: %s", club_name, team_id, exc)
            report["failed"] += 1
            continue

        if cached:
            report["cached"] += 1
        else:
            report["fetched"] += 1
            if budget is not None:
                budget -= 1
        report["players"] += len(rows)
        report["portraits"] += sum(1 for r in rows if r.get("photo"))

    refresh_caches()
    return report


def _match_team_id(club_name: str, entry: dict, team_ids: dict[str, int]) -> int | None:
    """Map a directory club onto an API-Football team id.

    The two providers name clubs differently ("FC Barcelona" against
    "Barcelona"), so this tries the exact name, the short name, and finally a
    normalised comparison — the same conservative ladder used elsewhere.
    """
    for candidate in (club_name, entry.get("short_name"), entry.get("name")):
        if candidate and candidate in team_ids:
            return team_ids[candidate]

    wanted = {normalise(c) for c in (club_name, entry.get("short_name")) if c}
    for name, team_id in team_ids.items():
        if normalise(name) in wanted:
            return team_id

    # Last resort: one side's normalised name is a leading run of the other's.
    for name, team_id in team_ids.items():
        key = normalise(name)
        for w in wanted:
            if len(key) >= 5 and (key.startswith(w) or w.startswith(key)):
                return team_id
    return None
