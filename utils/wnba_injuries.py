"""WNBA injury feed via ESPN's public API (no key required).

Mirrors the NBA injury path in utils/injury_news.py but stripped down: just
the structured API and per-team lookup. Cached in-memory for 30 minutes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NBAPredictor/1.0)"}
_ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"
_TTL_SECONDS = 30 * 60  # 30 min

# ESPN team display name -> tricode (backfill if `abbreviation` is null/missing)
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
    "Out": "OUT", "Day-To-Day": "QUESTIONABLE", "Doubtful": "DOUBTFUL",
    "Questionable": "QUESTIONABLE", "Probable": "PROBABLE",
    "Active": "ACTIVE",
}

_cache_by_team: dict[str, list[dict]] = {}
_cache_by_player: dict[str, dict] = {}
_cache_ts: Optional[datetime] = None


def _fetch_espn_wnba_injuries() -> None:
    """Populate module-level caches from ESPN. Silently no-ops on network failure."""
    global _cache_by_team, _cache_by_player, _cache_ts

    now = datetime.now()
    if _cache_ts and (now - _cache_ts).total_seconds() < _TTL_SECONDS:
        return

    try:
        resp = requests.get(_ESPN_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"[WNBA-Injury] ESPN fetch failed: {e}")
        return

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
            athlete = inj.get("athlete", {})
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
                "position": athlete.get("position", {}).get("abbreviation", ""),
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

    _cache_by_team = by_team
    _cache_by_player = by_player
    _cache_ts = now
    logger.info(f"[WNBA-Injury] Loaded {len(by_player)} players across {len(by_team)} teams")


def get_wnba_team_injuries(team_abbr: str) -> list[dict]:
    """Return non-active players for a team. Empty list if none / on fetch fail."""
    _fetch_espn_wnba_injuries()
    return _cache_by_team.get(team_abbr.upper(), [])


def get_wnba_player_injury(player_name: str) -> Optional[dict]:
    """Return injury entry for a specific player, or None if healthy/unknown."""
    _fetch_espn_wnba_injuries()
    return _cache_by_player.get(player_name.lower())


def get_all_wnba_injuries() -> dict[str, list[dict]]:
    """Return the full team → injuries map."""
    _fetch_espn_wnba_injuries()
    return dict(_cache_by_team)
