"""Every Division I player with real minutes, current and last season, for the ranked player board.

ESPN's ``byathlete`` feed sorts by minutes, so paging stops as soon as players fall below ``MIN_MPG``.
A finished season never changes, so it is fetched once and kept on disk.
"""

from __future__ import annotations

import logging
from typing import Any

from cbbpred import config as cfg
from cbbpred.context import Context
from cbbpred.feed import get_json
from cbbpred.store import read_json, write_json

log = logging.getLogger(__name__)

URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball/statistics/byathlete"
PAGE_SIZE = 500
MIN_MPG = 10.0
MAX_PAGES = 30            # safety stop; ~10k athletes exist in total
SEASONS_SHOWN = 2

# ESPN stat name -> short key in the export
STAT_KEYS = {
    "gamesPlayed": "gp", "avgMinutes": "min", "avgPoints": "pts", "avgRebounds": "reb", "avgAssists": "ast",
    "avgSteals": "stl", "avgBlocks": "blk", "avgTurnovers": "tov", "fieldGoalPct": "fg",
    "threePointFieldGoalPct": "tp", "freeThrowPct": "ft", "avgThreePointFieldGoalsMade": "tpm",
}


def _stats(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, float]:
    """Name -> value for one athlete row, aligning each category's ``values`` with the response's ``names``."""
    out: dict[str, float] = {}
    for head, cat in zip(payload.get("categories", []), row.get("categories", [])):
        out.update(zip(head.get("names", []), cat.get("values", [])))
    return out


def parse_row(payload: dict[str, Any], row: dict[str, Any], ctx: Context) -> dict[str, Any] | None:
    """One compact player record, or None when the player is not on a Division I team or has no stat line."""
    ath = row.get("athlete") or {}
    team_id = str(ath.get("teamId") or "")
    if not ath.get("id") or not ctx.is_d1(team_id):
        return None
    raw = _stats(payload, row)
    line = {short: raw[name] for name, short in STAT_KEYS.items() if isinstance(raw.get(name), (int, float))}
    if "min" not in line or "gp" not in line:
        return None
    team = ctx.teams.get(team_id, {})
    return {
        "id": str(ath["id"]), "name": ath.get("displayName"), "pos": (ath.get("position") or {}).get("abbreviation"),
        "team_id": team_id, "team": team.get("short") or team.get("name"), "abbr": team.get("abbr"),
        "logo": ctx.logo(team_id), "color": team.get("color"), "line": line,
    }


def fetch_season(season: int, ctx: Context) -> list[dict[str, Any]]:
    """Players for ``season`` with at least MIN_MPG minutes, most minutes first."""
    found: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        payload = get_json(URL, region="us", lang="en", contentorigin="espn", isqualified="false", page=page,
                           limit=PAGE_SIZE, season=season, seasontype=2, sort="general.avgMinutes:desc")
        rows = (payload or {}).get("athletes", [])
        found += [p for r in rows if (p := parse_row(payload, r, ctx)) and p["line"]["min"] >= MIN_MPG]
        lowest = _stats(payload, rows[-1]).get("avgMinutes", MIN_MPG) if rows else 0.0
        if not rows or page >= payload["pagination"]["pages"] or lowest < MIN_MPG:
            break
    return found


def season_players(season: int, current: int, ctx: Context) -> list[dict[str, Any]]:
    cache = cfg.data_dir() / f"players_{season}.json"
    if season < current and cache.exists():
        return read_json(cache)
    players = fetch_season(season, ctx)
    if season < current and players:
        write_json(players, cache, compact=True)
    return players


def merge(by_season: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One record per athlete: identity from the newest season they played, a stat line per season."""
    people: dict[str, dict[str, Any]] = {}
    for season in sorted(by_season):                      # oldest first so the newest overwrites identity
        for p in by_season[season]:
            base = {k: v for k, v in p.items() if k != "line"}
            person = people.setdefault(p["id"], {"stats": {}})
            person.update(base)
            person["stats"][str(season)] = p["line"]
    return sorted(people.values(), key=lambda p: -max(s.get("pts", 0) for s in p["stats"].values()))


def players_payload(ctx: Context, season: int) -> dict[str, Any]:
    seasons = [season - i for i in range(SEASONS_SHOWN)]
    by_season = {s: season_players(s, season, ctx) for s in seasons}
    return {"generated_at": ctx.generated_at.strftime("%Y-%m-%dT%H:%MZ"), "season": season, "seasons": seasons,
            "min_mpg": MIN_MPG, "players": merge(by_season)}
