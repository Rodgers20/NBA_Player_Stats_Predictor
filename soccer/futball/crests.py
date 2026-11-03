"""Club crests from TheSportsDB, cached on the T7.

TheSportsDB publishes club badges for free. Its bulk `lookup_all_teams`
endpoint is unreliable on the free key (it returns the same wrong 24 clubs for
every league id), so crests are resolved one club at a time via `searchteams`
and then cached permanently — a badge does not change.

Matching is verified, not assumed. football-data.co.uk writes "Nott'm Forest"
and "Ath Bilbao"; TheSportsDB writes "Nottingham Forest" and "Athletic Bilbao",
but helpfully carries a `strTeamAlternate` list that often contains the short
form verbatim. A candidate is accepted only if the canonical name matches the
club's name or one of its alternates after normalisation, or if an explicit
query override was used. Anything else is recorded as a miss rather than
guessed — a wrong crest is worse than no crest.
"""
from __future__ import annotations

import hashlib
import json
import struct
import logging
import threading
import time
import urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

from . import config
from .teams import normalise

log = logging.getLogger(__name__)

CACHE_DIR = config.CACHE / "crests"
MANIFEST = CACHE_DIR / "manifest.json"
USER_AGENT = "futball-dashboard/0.1 (https://github.com/futball-dashboard; personal analytics)"
SPORTSDB = "https://www.thesportsdb.com/api/v1/json/3"
WIKI_API = "https://en.wikipedia.org/w/api.php"
MISS_TTL = 7 * 24 * 3600

# Short forms that need a fuller query before TheSportsDB will find the club.
QUERY_OVERRIDES: dict[str, str] = {
    "Nott'm Forest": "Nottingham Forest",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Sheffield United": "Sheffield United",
    "Ath Bilbao": "Athletic Bilbao",
    "Ath Madrid": "Atletico Madrid",
    "Sociedad": "Real Sociedad",
    "Betis": "Real Betis",
    "Celta": "Celta Vigo",
    "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol",
    "Alaves": "Deportivo Alaves",
    "La Coruna": "Deportivo La Coruna",
    "Dep. A Coruna": "Deportivo La Coruna",
    "Santander": "Racing Santander",
    "Malaga": "Malaga",
    "Sp Gijon": "Sporting Gijon",
    "Vallodolid": "Real Valladolid",
    "Valladolid": "Real Valladolid",
    "M'gladbach": "Borussia Monchengladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Koln",
    "Paris SG": "Paris Saint-Germain",
    "St Etienne": "Saint-Etienne",
    "Nott": "Nottingham Forest",
    # short forms that TheSportsDB does not carry as an alternate
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "West Ham": "West Ham United",
    "Luton": "Luton Town",
    "Norwich": "Norwich City",
    "Leicester": "Leicester City",
    "Swansea": "Swansea City",
    "Cardiff": "Cardiff City",
    "Stoke": "Stoke City",
    "Coventry": "Coventry City",
    # UEFA long forms coming from openfootball
    "Athletic Club": "Athletic Bilbao",
    "Club Brugge KV": "Club Brugge",
    "FC København": "FC Copenhagen",
    "FK Bodø/Glimt": "Bodo Glimt",
    "FK Kairat": "Kairat",
    "Galatasaray SK": "Galatasaray",
    "Olympique de Marseille": "Marseille",
    "PAE Olympiakos SFP": "Olympiakos",
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "Qarabağ Ağdam FK": "Qarabag",
    "Royale Union Saint-Gilloise": "Union Saint-Gilloise",
    "SK Slavia Praha": "Slavia Prague",
    "Sport Lisboa e Benfica": "Benfica",
    "FC Internazionale Milano": "Inter Milan",
    "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "Club Atlético de Madrid": "Atletico Madrid",
    "Real Madrid CF": "Real Madrid",
    "Arsenal FC": "Arsenal",
    "Liverpool FC": "Liverpool",
    "Chelsea FC": "Chelsea",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "Newcastle United FC": "Newcastle United",
    "Villarreal CF": "Villarreal",
    "Juventus FC": "Juventus",
    "Atalanta BC": "Atalanta",
    "AS Monaco FC": "Monaco",
    "PSV": "PSV Eindhoven",
    "Paphos FC": "Pafos",
    "AFC Ajax": "Ajax",
    "Brighton": "Brighton and Hove Albion",
}

_lock = threading.Lock()
_manifest: dict[str, dict] | None = None


@dataclass
class Crest:
    name: str
    found: bool
    source: str | None = None
    image_url: str | None = None
    matched_name: str | None = None
    file: str | None = None
    checked_at: float = 0.0


def _load() -> dict[str, dict]:
    global _manifest
    if _manifest is None:
        try:
            _manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        except Exception:  # noqa: BLE001
            _manifest = {}
    return _manifest


def _save() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(_load(), indent=1))
    tmp.replace(MANIFEST)


def _key(name: str) -> str:
    return normalise(name) or (name or "").strip().lower()


def _slug(name: str) -> str:
    return hashlib.sha256(_key(name).encode()).hexdigest()[:20]


def _client() -> httpx.Client:
    return httpx.Client(timeout=25.0, headers={"User-Agent": USER_AGENT},
                        follow_redirects=True)


def _alternates(team: dict) -> set[str]:
    out = {normalise(team.get("strTeam") or "")}
    alt = team.get("strTeamAlternate") or ""
    for piece in alt.split(","):
        n = normalise(piece)
        if n:
            out.add(n)
    return {o for o in out if o}


def _sportsdb(client: httpx.Client, canonical: str) -> tuple[str | None, str | None]:
    """(badge_url, matched_name) or (None, None).

    The free tier rate-limits and briefly answers with non-JSON, so a failed
    parse is retried rather than recorded as "this club has no badge" — that
    mistake poisons the cache with permanent misses.
    """
    query = QUERY_OVERRIDES.get(canonical, canonical)
    teams: list[dict] = []
    for attempt in range(4):
        try:
            r = client.get(f"{SPORTSDB}/searchteams.php", params={"t": query})
            if r.status_code == 200:
                teams = (r.json() or {}).get("teams") or []
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5 * (attempt + 1))
    else:
        return None, None

    want = {normalise(canonical), normalise(query)}
    for t in teams:
        if (t.get("strSport") or "Soccer") != "Soccer":
            continue
        badge = t.get("strBadge") or t.get("strTeamBadge")
        if not badge:
            continue
        if _alternates(t) & want:
            return badge, t.get("strTeam")
    # An override is an explicit statement of intent, so the first soccer club
    # returned for it is accepted even if the alternates do not line up.
    if canonical in QUERY_OVERRIDES:
        for t in teams:
            badge = t.get("strBadge") or t.get("strTeamBadge")
            if badge and (t.get("strSport") or "Soccer") == "Soccer":
                return badge, t.get("strTeam")
    return None, None


def _wikipedia(client: httpx.Client, canonical: str) -> tuple[str | None, str | None]:
    query = QUERY_OVERRIDES.get(canonical, canonical)
    try:
        r = client.get(WIKI_API, params={
            "action": "query", "format": "json", "prop": "pageimages",
            "piprop": "thumbnail", "pithumbsize": 160,
            "titles": f"{query} F.C.", "redirects": 1,
        })
        if r.status_code == 200:
            for pid, page in (r.json().get("query", {}).get("pages", {}) or {}).items():
                if str(pid) == "-1":
                    continue
                thumb = (page.get("thumbnail") or {}).get("source")
                if thumb:
                    return thumb, page.get("title")
    except Exception:  # noqa: BLE001
        pass
    return None, None


def _aspect_ok(data: bytes) -> bool:
    """A club crest is roughly square; a banner or wordmark is not.

    Wikipedia's fallback once returned a 250x90 club banner for Brighton, which
    letterboxed into a thin strip in every table. Checking the intrinsic size
    rejects that shape rather than rendering it badly.
    """
    w = h = None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
    elif data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                break
            seg = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + seg
    if not w or not h:
        return True  # unknown format (e.g. svg/webp) — accept
    ratio = w / h
    return 0.6 <= ratio <= 1.7


def _download(client: httpx.Client, url: str, name: str) -> str | None:
    try:
        r = client.get(url)
        if r.status_code != 200 or not r.content:
            return None
    except Exception:  # noqa: BLE001
        return None
    if not _aspect_ok(r.content):
        log.info("rejecting non-square crest for %s (%s)", name, url)
        return None
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"):
        ext = ".png"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{_slug(name)}{ext}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return dest.name


def resolve(name: str, *, download: bool = True) -> Crest:
    key = _key(name)
    if not key:
        return Crest(name=name, found=False)

    with _lock:
        cached = _load().get(key)
    if cached:
        c = Crest(**cached)
        fresh = c.found or (time.time() - c.checked_at) < MISS_TTL
        has_file = bool(c.file) and (CACHE_DIR / c.file).exists()
        if fresh and (not c.found or has_file or not download):
            return c

    with _client() as client:
        url, matched = _sportsdb(client, name)
        source = "thesportsdb" if url else None
        if not url:
            url, matched = _wikipedia(client, name)
            source = "wikipedia" if url else None
        file_name = _download(client, url, name) if (url and download) else None

    c = Crest(name=name, found=bool(url), source=source, image_url=url,
              matched_name=matched, file=file_name, checked_at=time.time())
    with _lock:
        _load()[key] = asdict(c)
        _save()
    return c


def cached_path(name: str) -> Path | None:
    with _lock:
        rec = _load().get(_key(name))
    if not rec or not rec.get("file"):
        return None
    p = CACHE_DIR / rec["file"]
    return p if p.exists() else None


def warm(names: list[str], pause: float = 0.35) -> dict:
    found = missed = 0
    misses: list[str] = []
    for n in names:
        c = resolve(n)
        if c.found:
            found += 1
        else:
            missed += 1
            misses.append(n)
        time.sleep(pause)
    return {"found": found, "missing": missed, "misses": misses}


def stats() -> dict:
    m = _load()
    return {"names": len(m), "found": sum(1 for v in m.values() if v.get("found"))}
