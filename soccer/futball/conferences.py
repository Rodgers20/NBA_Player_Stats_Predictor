"""Conference tables for leagues that split in two (MLS): membership from ESPN, ranking within each conference.

The MLS Cup playoffs take the top nine of each conference, so a single combined table cannot show the playoff line.
Membership comes from ESPN's standings feed and is cached for six hours; if it is unavailable the caller falls back to
one combined table with no playoff band, because a wrong band is worse than none.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from . import boxscores, config
from .sources import _cache

log = logging.getLogger(__name__)

CONFERENCE_LEAGUES: tuple[str, ...] = ("MLS",)
TTL_SECONDS = 6 * 3600
ORDER = ("Eastern Conference", "Western Conference")


def parse(body: dict, resolve: Callable[[str, str], str] = boxscores.resolve_team, code: str = "MLS") -> dict[str, str]:
    """team (lake spelling) -> conference name."""
    out: dict[str, str] = {}
    for child in (body or {}).get("children") or []:
        name = child.get("name")
        for entry in ((child.get("standings") or {}).get("entries")) or []:
            team = (entry.get("team") or {}).get("displayName")
            if name and team:
                out[resolve(team, code)] = name
    return out


def conferences(code: str) -> dict[str, str]:
    """Membership for `code`, from cache or ESPN. {} when the feed is unavailable."""
    if code not in CONFERENCE_LEAGUES:
        return {}
    path = config.RAW / "espn-box" / code / "standings.json"
    if not _cache.is_fresh(path, TTL_SECONDS):
        try:
            body = boxscores._get(f"https://site.api.espn.com/apis/v2/sports/soccer/{config.ESPN_SLUGS[code]}/standings")
            _cache.write_json(path, body)
        except RuntimeError as exc:
            log.warning("ESPN conference standings unavailable (%s)", exc)
    return parse(_cache.read_json(path) or {}, code=code)


def rank_in_conferences(table: list[dict], membership: dict[str, str],
                        band: Callable[[int], tuple[str, str] | None]) -> list[dict]:
    """Re-rank an overall points-ordered table within each conference and attach the playoff band.

    Rows are returned grouped (Eastern, Western, then any club with no known conference), each group in table order.
    """
    groups: dict[str | None, list[dict]] = {name: [] for name in ORDER}
    groups[None] = []
    for row in table:
        conference = membership.get(row["team"])
        groups.setdefault(conference if conference in groups else None, []).append(row)
    overall = {row["team"]: i for i, row in enumerate(table, 1)}
    out: list[dict] = []
    for conference, members in groups.items():
        for rank, row in enumerate(members, 1):
            position = rank if conference else overall[row["team"]]  # unplaced clubs keep their overall position
            info = band(position) if conference else None
            out.append({**row, "conference": conference, "position": position,
                        "qualification": info[0] if info else None, "qualification_tone": info[1] if info else None})
    return out
