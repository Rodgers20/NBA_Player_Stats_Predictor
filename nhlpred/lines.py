"""Game lines (moneyline, puck line, total) from ESPN's free event summaries.

ESPN carries game odds for NHL but no player props. Pre-game lines are re-read on every run so movement
shows up until puck drop; once a game has started its last pre-game line is kept and never overwritten
by live odds. ``spread`` follows ESPN: the HOME team's puck line (negative = home favored).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from nhlpred import config as cfg
from nhlpred.feed import MAX_WORKERS, FeedError, get_json
from nhlpred.oddsmath import american_to_prob
from nhlpred.store import read_json, write_json

log = logging.getLogger(__name__)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
ALIAS = {"SJ": "SJS", "TB": "TBL", "NJ": "NJD", "LA": "LAK", "UTAH": "UTA"}   # ESPN abbreviation -> NHL API abbreviation
KEEP_DAYS = 14


def lines_path():
    return cfg.data_dir() / "lines.json"


def _num(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _abbr(a: str | None) -> str:
    a = (a or "").upper()
    return ALIAS.get(a, a)


def parse_pickcenter(summary: dict[str, Any], fetched_at: str) -> dict[str, Any] | None:
    """The first sportsbook's moneyline, puck line and total, or None when no usable price is posted."""
    for item in summary.get("pickcenter") or []:
        home, away = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}
        row = {"book": (item.get("provider") or {}).get("name"), "home_ml": _num(home.get("moneyLine")), "away_ml": _num(away.get("moneyLine")),
               "total": _num(item.get("overUnder")), "over_odds": _num(item.get("overOdds")), "under_odds": _num(item.get("underOdds")),
               "puck_home": _num(item.get("spread")), "home_spread_odds": _num(home.get("spreadOdds")), "away_spread_odds": _num(away.get("spreadOdds")),
               "fetched_at": fetched_at}
        if row["home_ml"] is not None and row["away_ml"] is not None:
            return row
    return None


def no_vig_home(lines: dict[str, Any] | None) -> float | None:
    """Market win probability of the home team with the book's margin removed."""
    if not lines:
        return None
    h, a = american_to_prob(lines.get("home_ml")), american_to_prob(lines.get("away_ml"))
    return None if h is None or a is None else h / (h + a)


def _events(dates: list[str]) -> dict[tuple[str, str, str], str]:
    """(date, home, away) -> ESPN event id; the scoreboard takes one day per request."""
    out = {}
    for day in dates:
        d = get_json(f"{ESPN}/scoreboard?dates={day.replace('-', '')}") or {}
        for e in d.get("events", []):
            comp = (e.get("competitions") or [{}])[0]
            side = {c.get("homeAway"): _abbr((c.get("team") or {}).get("abbreviation")) for c in comp.get("competitors", [])}
            if side.get("home") and side.get("away"):
                out[(day, side["home"], side["away"])] = e["id"]
    return out


def refresh_lines(games: pd.DataFrame, now: datetime | None = None) -> dict[str, Any]:
    """Update ``data/nhl/lines.json`` for games that have not started; returns the stored mapping."""
    now = now or datetime.now(UTC)
    stored: dict[str, Any] = read_json(lines_path(), {})
    cutoff = (now - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    stored = {k: v for k, v in stored.items() if v.get("date", "9999") >= cutoff}
    todo = games[(games.status == "scheduled") & (pd.to_datetime(games.start_utc, utc=True) > now)
                 & (games.date <= (now + timedelta(days=cfg.EXPORT_DAYS_AHEAD)).strftime("%Y-%m-%d"))]
    if todo.empty:
        write_json(stored, lines_path())
        return stored
    try:
        events = _events(sorted(todo.date.unique()))
    except FeedError as exc:
        log.warning("ESPN schedule unavailable, keeping stored lines: %s", exc)
        return stored

    def one(g: pd.Series) -> tuple[str, dict[str, Any] | None]:
        eid = events.get((g.date, _abbr(g.home_abbr), _abbr(g.away_abbr)))
        if eid is None:
            return str(int(g.game_id)), None
        try:
            return str(int(g.game_id)), parse_pickcenter(get_json(f"{ESPN}/summary?event={eid}") or {}, now.strftime("%Y-%m-%dT%H:%MZ"))
        except FeedError as exc:
            log.warning("no odds for %s @ %s: %s", g.away_abbr, g.home_abbr, exc)
            return str(int(g.game_id)), None

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        for gid, row in pool.map(one, [g for _, g in todo.iterrows()]):
            if row:
                game_date = str(todo[todo.game_id == int(gid)].date.iloc[0])
                prev = stored.get(gid, {})
                stored[gid] = {**row, "date": game_date, "open": prev.get("open") or {k: row[k] for k in ("home_ml", "away_ml", "total", "puck_home")}}
    write_json(stored, lines_path())
    log.info("lines for %s of %s upcoming games", sum(str(int(g)) in stored for g in todo.game_id), len(todo))
    return stored


def ev_per_unit(p: float | None, odds: float | None) -> float | None:
    """Expected profit per 1 staked when the true win chance is ``p`` and the book pays American ``odds``."""
    if p is None or odds in (None, 0):
        return None
    return p * (odds / 100 if odds > 0 else 100 / -odds) - (1 - p)
