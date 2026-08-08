"""WNBA injury feed with layered fallback.

Fetch order:
  1. ESPN web API (`site.web.api.espn.com`) — currently works
  2. ESPN legacy API (`site.api.espn.com`) — blocked as of Aug 2026 but kept as fallback
  3. Disk cache (data/wnba/injuries_cache.json) — last-good data if both fail

Player lookup is fuzzy-tolerant: exact match → last-name + first-initial
→ last-name only, so "Skylar Diggins" finds "Skylar Diggins-Smith" and vice versa.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from utils.league_config import get_config

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.espn.com/wnba/injuries",
    "Origin": "https://www.espn.com",
}

# Endpoints in fetch priority order
_ESPN_ENDPOINTS = [
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries",
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries",
]

_CACHE_FILE = get_config("wnba").data_dir / "injuries_cache.json"
_TTL_SECONDS = 30 * 60  # 30 min in-memory TTL
_DISK_TTL_HOURS = 24    # accept disk cache up to 24h old

_TEAM_NAME_TO_ABBR = {
    "Atlanta Dream": "ATL", "Chicago Sky": "CHI", "Connecticut Sun": "CON",
    "Dallas Wings": "DAL", "Golden State Valkyries": "GSV",
    "Indiana Fever": "IND", "Las Vegas Aces": "LVA",
    "Los Angeles Sparks": "LAS", "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL", "Portland Fire": "PDX",
    "Phoenix Mercury": "PHX", "Seattle Storm": "SEA",
    "Toronto Tempo": "TOR", "Washington Mystics": "WAS",
}

_ESPN_STATUS_MAP = {
    "Out": "OUT",
    "Out for Season": "OUT_SEASON",
    "Suspended": "OUT",
    "Day-To-Day": "QUESTIONABLE",
    "Doubtful": "DOUBTFUL",
    "Questionable": "QUESTIONABLE",
    "Probable": "PROBABLE",
    "Active": "ACTIVE",
}

# Statuses that mean the player is NOT going to play tonight
_UNAVAILABLE_STATUSES = {"OUT", "OUT_SEASON", "DOUBTFUL"}


_cache_by_team: dict[str, list[dict]] = {}
_cache_by_player: dict[str, dict] = {}
_cache_ts: Optional[datetime] = None


# ── Fetch ───────────────────────────────────────────────────────────────────

def _fetch_from_espn() -> Optional[dict]:
    """Try each ESPN endpoint in order. Return first successful JSON, or None."""
    for url in _ESPN_ENDPOINTS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            logger.debug(f"[WNBA-Injury] {url} -> {resp.status_code}")
        except Exception as e:
            logger.debug(f"[WNBA-Injury] {url} error: {e}")
    return None


def _load_disk_cache() -> Optional[dict]:
    """Read the last-good injury payload from disk if it's not too stale."""
    if not _CACHE_FILE.exists():
        return None
    try:
        blob = json.loads(_CACHE_FILE.read_text())
        cached_at = datetime.fromisoformat(blob["cached_at"])
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600
        if age_hours > _DISK_TTL_HOURS:
            logger.info(f"[WNBA-Injury] disk cache is {age_hours:.1f}h old — considered stale")
            return None
        logger.info(f"[WNBA-Injury] using disk cache ({age_hours:.1f}h old)")
        return blob["payload"]
    except Exception as e:
        logger.warning(f"[WNBA-Injury] disk cache read failed: {e}")
        return None


def _save_disk_cache(payload: dict) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({
            "cached_at": datetime.now().isoformat(),
            "payload": payload,
        }, indent=2))
    except Exception as e:
        logger.warning(f"[WNBA-Injury] disk cache write failed: {e}")


# ── Parse ────────────────────────────────────────────────────────────────────

def _parse_payload(data: dict) -> tuple[dict, dict]:
    """Split ESPN payload into (by_team, by_player_lower) dicts."""
    by_team: dict[str, list[dict]] = {}
    by_player: dict[str, dict] = {}

    for team_block in data.get("injuries", []):
        team_name = team_block.get("displayName", "")
        abbrev = (
            team_block.get("abbreviation")
            or _TEAM_NAME_TO_ABBR.get(team_name)
            or team_name[:3].upper()
        )
        team_list = []
        for inj in team_block.get("injuries", []):
            athlete = inj.get("athlete") or {}
            name = athlete.get("displayName", "")
            if not name:
                continue
            raw_status = inj.get("status", "Active")
            mapped = _ESPN_STATUS_MAP.get(raw_status, "OUT")
            reason = (inj.get("shortComment") or inj.get("longComment") or raw_status).strip()

            entry = {
                "name": name,
                "team_abbr": abbrev,
                "status": mapped,
                "reason": reason,
                "position": (athlete.get("position") or {}).get("abbreviation", ""),
            }
            by_player[name.lower()] = entry
            if mapped != "ACTIVE":
                team_list.append({
                    "name": name,
                    "status": mapped,
                    "reason": reason,
                    "position": entry["position"],
                })

        if team_list:
            by_team[abbrev] = team_list

    return by_team, by_player


def _refresh() -> None:
    """Refresh module-level caches from ESPN, falling back to disk if needed."""
    global _cache_by_team, _cache_by_player, _cache_ts

    now = datetime.now()
    if _cache_ts and (now - _cache_ts).total_seconds() < _TTL_SECONDS:
        return

    payload = _fetch_from_espn()
    if payload is None:
        payload = _load_disk_cache()

    if payload is None:
        logger.warning("[WNBA-Injury] all sources failed — cache stays empty")
        return

    by_team, by_player = _parse_payload(payload)
    _cache_by_team = by_team
    _cache_by_player = by_player
    _cache_ts = now
    _save_disk_cache(payload)
    logger.info(f"[WNBA-Injury] Loaded {len(by_player)} players across {len(by_team)} teams")


# ── Fuzzy lookup ─────────────────────────────────────────────────────────────

def _fuzzy_player_lookup(name: str) -> Optional[dict]:
    """Exact → last+first-initial → last-name-only, tolerant of hyphenated
    last names ("Skylar Diggins" ↔ "Skylar Diggins-Smith")."""
    key = name.lower().strip()
    hit = _cache_by_player.get(key)
    if hit:
        return hit

    parts = key.split()
    if not parts:
        return None
    # A hyphenated last name → treat each component as a valid last-name candidate
    last_candidates: list[str] = []
    for chunk in parts[-1].split("-"):
        if chunk:
            last_candidates.append(chunk)
    if not last_candidates:
        return None
    first_initial = parts[0][:1] if parts else ""

    # Also normalize each cache key's last-name into candidate set
    def _last_candidates_for(k: str) -> list[str]:
        p = k.split()
        if not p:
            return []
        return [c for c in p[-1].split("-") if c]

    # Last + first-initial (any candidate overlap)
    matches = []
    for k, v in _cache_by_player.items():
        cand = _last_candidates_for(k)
        if not cand:
            continue
        k_first_init = k.split()[0][:1]
        if k_first_init == first_initial and any(c in last_candidates for c in cand):
            matches.append(v)
    if len(matches) == 1:
        return matches[0]

    # Last-name only — must be unique
    last_only = []
    for k, v in _cache_by_player.items():
        cand = _last_candidates_for(k)
        if any(c in last_candidates for c in cand):
            last_only.append(v)
    if len(last_only) == 1:
        return last_only[0]

    return None


# ── Public API ───────────────────────────────────────────────────────────────

def get_wnba_team_injuries(team_abbr: str) -> list[dict]:
    _refresh()
    return _cache_by_team.get(team_abbr.upper(), [])


def get_wnba_player_injury(player_name: str) -> Optional[dict]:
    """Return injury entry for a player. Fuzzy-tolerant on the name."""
    _refresh()
    return _fuzzy_player_lookup(player_name)


def get_all_wnba_injuries() -> dict[str, list[dict]]:
    _refresh()
    return dict(_cache_by_team)


def is_player_unavailable(player_name: str) -> bool:
    """True if the player is OUT / OUT_SEASON / DOUBTFUL. Used to filter props."""
    inj = get_wnba_player_injury(player_name)
    if not inj:
        return False
    return inj.get("status") in _UNAVAILABLE_STATUSES
