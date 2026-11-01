"""ESPN men's college basketball feed: games by date, teams, conferences, AP poll.

Everything is free and keyless. Seasons are named by the year they end in. Scoreboard rows carry no
sportsbook lines for college; those come from ``cbbpred.lines``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from cbbpred import config as cfg
from cbbpred.feed import MAX_WORKERS, FeedError, get_json
from cbbpred.store import read_json, read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
STANDINGS_URL = "https://site.web.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/standings"
EASTERN = ZoneInfo("America/New_York")
SCHEDULE_AHEAD_DAYS = 14      # how far past today a refresh asks for the schedule
REFRESH_DAYS_BACK = 5         # a refresh re-reads this many past days to pick up final scores
REGULATION_PERIODS = 2
OFF_STATUSES = ("POSTPONED", "CANCELED", "CANCELLED", "SUSPENDED", "FORFEIT")

GAME_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "season_type": pl.Int64,
    "kickoff": pl.Datetime("us", "UTC"), "gameday": pl.Utf8, "gametime": pl.Utf8,
    "status": pl.Utf8, "overtimes": pl.Int64, "neutral": pl.Boolean, "conf_game": pl.Boolean,
    "note": pl.Utf8, "venue": pl.Utf8, "broadcast": pl.Utf8,
    **{f"{s}_{k}": t for s in ("home", "away") for k, t in (
        ("id", pl.Utf8), ("name", pl.Utf8), ("short", pl.Utf8), ("abbr", pl.Utf8), ("conf", pl.Utf8),
        ("rank", pl.Int64), ("score", pl.Int64))},
}


def current_season(now: datetime | None = None) -> int:
    """The season in play: it starts in November and ends the following April."""
    now = now or datetime.now(UTC)
    return now.year + 1 if now.month >= 7 else now.year


def season_dates(season: int, today: date) -> list[date]:
    """Every date to scan for a season; the running season stops SCHEDULE_AHEAD_DAYS past today."""
    start = date(season - 1, *cfg.SEASON_OPEN)
    end = min(date(season, *cfg.SEASON_CLOSE), today + timedelta(days=SCHEDULE_AHEAD_DAYS))
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _status(comp: dict[str, Any]) -> str:
    kind = comp["status"]["type"]
    if any(w in kind["name"].upper() for w in OFF_STATUSES):
        return "postponed"
    if kind.get("completed"):
        return "final"
    return {"pre": "scheduled", "in": "live"}.get(kind.get("state"), "other")


def _rank(competitor: dict[str, Any]) -> int | None:
    current = (competitor.get("curatedRank") or {}).get("current")
    return current if isinstance(current, int) and 1 <= current <= cfg.POLL_SIZE else None


def _score(competitor: dict[str, Any], status: str) -> int | None:
    if status in ("scheduled", "postponed"):
        return None
    try:
        return int(competitor["score"])
    except (KeyError, TypeError, ValueError):
        return None


def parse_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """One game row, or None for placeholder teams, preseason exhibitions and malformed events."""
    if event["season"]["type"] not in cfg.SEASON_TYPES:
        return None
    comp = event["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    if set(sides) != {"home", "away"}:
        return None
    if any(str(c["team"].get("id", "-1")).startswith("-") or c["team"].get("abbreviation") == "TBD"
           for c in sides.values()):
        return None
    status = _status(comp)
    kickoff = datetime.strptime(event["date"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
    local = kickoff.astimezone(EASTERN)
    notes = comp.get("notes") or []
    period = comp["status"].get("period") or 0
    row: dict[str, Any] = {
        "game_id": str(event["id"]), "season": event["season"]["year"], "season_type": event["season"]["type"],
        "kickoff": kickoff, "gameday": local.strftime("%Y-%m-%d"), "gametime": local.strftime("%H:%M"),
        "status": status, "overtimes": max(0, period - REGULATION_PERIODS) if status == "final" else 0,
        "neutral": bool(comp.get("neutralSite")), "conf_game": bool(comp.get("conferenceCompetition")),
        "note": notes[0].get("headline") if notes else None,
        "venue": (comp.get("venue") or {}).get("fullName"), "broadcast": comp.get("broadcast"),
    }
    for side, c in sides.items():
        team = c["team"]
        conf = team.get("conferenceId")
        row |= {
            f"{side}_id": str(team["id"]), f"{side}_name": team.get("displayName"),
            f"{side}_short": team.get("shortDisplayName"), f"{side}_abbr": team.get("abbreviation"),
            f"{side}_conf": str(conf) if conf not in (None, "") else None,
            f"{side}_rank": _rank(c), f"{side}_score": _score(c, status),
        }
    return row


def fetch_day(day: date) -> list[dict[str, Any]]:
    payload = get_json(f"{BASE}/scoreboard", dates=day.strftime("%Y%m%d"), groups=cfg.D1_GROUP, limit=400)
    if payload is None or "events" not in payload:
        raise FeedError(f"ESPN scoreboard for {day} has no 'events' key; the format changed?")
    return payload["events"]


def fetch_games(days: list[date]) -> tuple[list[dict[str, Any]], list[date]]:
    """Parsed game rows for ``days`` plus the days that failed after retries."""
    def one(day: date) -> tuple[date, list[dict[str, Any]] | None]:
        try:
            return day, [r for ev in fetch_day(day) if (r := parse_event(ev)) is not None]
        except FeedError as exc:
            log.warning("scoreboard %s failed: %s", day, exc)
            return day, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(one, days))
    rows = [r for _, batch in results if batch for r in batch]
    return rows, [d for d, batch in results if batch is None]


def merge_games(new: pl.DataFrame, window: tuple[date, date] | None = None) -> pl.DataFrame:
    """Fold fresh rows into games.parquet; a game id keeps its newest row.

    A game still marked scheduled inside the refreshed ``window`` that ESPN no longer lists was moved or
    dropped, so it goes too; otherwise it would sit in the schedule forever.
    """
    old = read_parquet(cfg.games_path())
    if old.height and window:
        lo, hi = (window[0] + timedelta(days=1)).isoformat(), (window[1] - timedelta(days=1)).isoformat()
        gone = (pl.col("status") == "scheduled") & pl.col("gameday").is_between(pl.lit(lo), pl.lit(hi)) \
            & ~pl.col("game_id").is_in(new["game_id"].implode())
        old = old.filter(~gone)
    merged = pl.concat([old, new], how="diagonal_relaxed") if old.height else new
    merged = merged.unique(subset="game_id", keep="last", maintain_order=True).sort("kickoff", "game_id")
    write_parquet(merged, cfg.games_path())
    return merged


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=GAME_SCHEMA)


def stale_start(today: date) -> date:
    """Where a refresh starts: a few days back, or earlier if a played game never got its final score."""
    start = today - timedelta(days=REFRESH_DAYS_BACK)
    games = read_parquet(cfg.games_path())
    if games.is_empty():
        return start
    floor = (today - timedelta(days=cfg.STALE_LOOKBACK_DAYS)).isoformat()
    open_ = games.filter(pl.col("status").is_in(["scheduled", "live"])
                         & pl.col("gameday").is_between(pl.lit(floor), pl.lit((today - timedelta(days=1)).isoformat())))
    if open_.is_empty():
        return start
    return min(start, date.fromisoformat(open_["gameday"].min()))


def ingest_games(seasons: list[int], *, refresh: bool, today: date | None = None) -> pl.DataFrame:
    """Pull ``seasons`` whole, or with ``refresh`` only the recent/near-future dates of the running season."""
    today = today or datetime.now(EASTERN).date()
    games = read_parquet(cfg.games_path())
    failed: list[date] = []
    first_day = stale_start(today) if refresh else None
    for season in seasons:
        days = season_dates(season, today)
        if first_day is not None:
            days = [d for d in days if first_day <= d]
        if not days:
            continue
        rows, bad = fetch_games(days)
        failed += bad
        if rows:
            games = merge_games(_frame(rows), (days[0], days[-1]) if refresh and not bad else None)
        log.info("season %s: %s games from %s dates (%s failed)", season, len(rows), len(days), len(bad))
    if failed:
        raise FeedError(f"{len(failed)} scoreboard dates failed (first {failed[0]}); rerun to fill them")
    return games


def fetch_teams() -> pl.DataFrame:
    payload = get_json(f"{BASE}/teams", limit=1000)
    teams = payload["sports"][0]["leagues"][0]["teams"]
    return pl.DataFrame([{
        "team_id": str(t["team"]["id"]), "name": t["team"].get("displayName"),
        "short": t["team"].get("shortDisplayName"), "abbr": t["team"].get("abbreviation"),
        "color": t["team"].get("color"), "alt_color": t["team"].get("alternateColor"),
        "logo": next(iter(t["team"].get("logos") or []), {}).get("href"),
    } for t in teams])


def fetch_conferences() -> dict[str, dict[str, Any]]:
    """{group id: {name, short, logo}} for the Division I conferences."""
    groups = get_json(f"{BASE}/scoreboard/conferences")["conferences"]
    found = {g["groupId"]: {"name": g["name"], "short": g["shortName"], "logo": g.get("logo")}
             for g in groups if g.get("parentGroupId") == cfg.D1_GROUP}
    if not found:
        # An empty list would make every team look non-D1 and quietly skew every rating.
        raise FeedError("ESPN listed no Division I conferences; the format changed?")
    return found


def fetch_team_conferences(season: int) -> dict[str, str]:
    """{team id: conference group id} from ESPN's standings, which list teams before they have played."""
    payload = get_json(STANDINGS_URL, season=season)
    out = {str(e["team"]["id"]): c["id"] for c in payload.get("children", [])
           for e in (c.get("standings") or {}).get("entries", [])}
    if not out:
        raise FeedError(f"ESPN standings for {season} list no teams; the format changed?")
    return out


def fetch_ap_poll() -> dict[str, Any]:
    """The current AP Top 25: {week, label, updated, ranks: {team id: {rank, previous, points, ...}}}."""
    payload = get_json(f"{BASE}/rankings")
    ap = next((r for r in payload.get("rankings", []) if r.get("type") == "ap"), None)
    if ap is None:
        raise FeedError("ESPN rankings payload has no AP poll; the format changed?")
    week = payload.get("latestWeek") or {}
    return {"week": week.get("number"), "label": week.get("displayValue"), "updated": ap.get("lastUpdated"),
            "ranks": {str(r["team"]["id"]): {"rank": r["current"], "previous": r.get("previous"),
                                              "points": r.get("points"), "first_place": r.get("firstPlaceVotes"),
                                              "record": r.get("recordSummary")}
                      for r in ap.get("ranks", [])}}


def _refresh_reference(label: str, fetch: Callable[[], Any], write: Callable[[Any], None]) -> None:
    """Reference tables describe the games, which are already saved: a feed that fails keeps the last copy."""
    try:
        write(fetch())
    except FeedError as exc:
        log.warning("%s not refreshed, keeping the last copy: %s", label, exc)


def refresh_reference(season: int) -> None:
    _refresh_reference("teams", fetch_teams, lambda df: write_parquet(df, cfg.teams_path()))
    _refresh_reference("conferences", fetch_conferences, lambda d: write_json(d, cfg.conferences_path()))
    _refresh_reference("team conferences", lambda: fetch_team_conferences(season),
                       lambda d: write_json(d, cfg.team_conf_path()))
    _refresh_reference("AP poll", fetch_ap_poll, lambda d: write_json(d, cfg.ap_poll_path()))
    if not cfg.conferences_path().exists():
        raise FeedError("no conference list on disk; ratings need it to tell Division I from the rest")


def conferences() -> dict[str, dict[str, Any]]:
    return read_json(cfg.conferences_path()) if cfg.conferences_path().exists() else {}
