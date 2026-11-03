"""ESPN box-score backfill: scoreboard days -> finished events -> summaries -> `player_match_stats`.

ESPN's free summary endpoint carries a per-player stat line for every finished match
(appearances, goals, assists, shots, shots on target, fouls, cards, offsides, saves) but
no minutes and no tackles. Everything is cached under `raw/espn-box/` and the cache is the
resume point: a finished match is fetched once, ever; a scoreboard day is final once it is
more than a day old. ESPN answers 400 to multi-day `dates` ranges, so every request is a
single day.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable

import httpx
import polars as pl

from . import config
from .sources import _cache
from .storage import atomic_path

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
RAW = config.RAW / "espn-box"
SILVER_DIR = config.SILVER / "player_stats"
STATS_FILE = SILVER_DIR / "player_match_stats.parquet"
MIN_INTERVAL = 0.12  # seconds between requests per worker; ESPN is generous but this is a shared courtesy

# our column -> ESPN stat name
STAT_FIELDS: dict[str, str] = {
    "goals": "totalGoals", "assists": "goalAssists", "shots": "totalShots", "sot": "shotsOnTarget",
    "fouls_committed": "foulsCommitted", "fouls_suffered": "foulsSuffered", "yellow": "yellowCards",
    "red": "redCards", "offsides": "offsides", "saves": "saves",
}
GROUP_LETTER = {"GK": "G", "DEF": "D", "MID": "M", "ATT": "F"}

SCHEMA: dict[str, pl.DataType] = {
    "espn_id": pl.Utf8, "player": pl.Utf8, "team": pl.Utf8, "opponent": pl.Utf8,
    "competition_code": pl.Utf8, "date": pl.Utf8, "event_id": pl.Utf8, "is_home": pl.Boolean,
    "started": pl.Boolean, "sub_in": pl.Boolean, "position": pl.Utf8,
    "goals": pl.Int32, "assists": pl.Int32, "shots": pl.Int32, "sot": pl.Int32,
    "fouls_committed": pl.Int32, "fouls_suffered": pl.Int32, "yellow": pl.Int32, "red": pl.Int32,
    "offsides": pl.Int32, "saves": pl.Int32, "team_goals": pl.Int32, "opp_goals": pl.Int32,
}


def season_start(code: str, today: date | None = None) -> date:
    """First day of the season in progress: 1 Aug for Europe, 1 Jan for MLS (calendar-year season)."""
    today = today or date.today()
    if code == "MLS":
        return date(today.year, 1, 1)
    return date(today.year if today.month >= 7 else today.year - 1, 8, 1)


# ---------------------------------------------------------------- parsing ---

def _int(value: object) -> int:
    """ESPN sends stat values as floats and scoreboard scores as strings; anything else is 0."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return 0


def _is_exhibition(meta: dict) -> bool:
    """All-Star games share the league's scoreboard but are not league matches."""
    return "all-star" in f"{meta.get('home', '')} {meta.get('away', '')}".lower()


def completed_events(board: dict) -> list[dict]:
    """Finished events on a scoreboard, reduced to the fields the backfill needs."""
    out: list[dict] = []
    for event in (board or {}).get("events") or []:
        status = ((event.get("status") or {}).get("type") or {})
        competition = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in competition.get("competitors") or []}
        if not status.get("completed") or "home" not in sides or "away" not in sides:
            continue
        home = sides["home"].get("team", {}).get("displayName", "")
        away = sides["away"].get("team", {}).get("displayName", "")
        if _is_exhibition({"home": home, "away": away}):
            continue
        out.append({
            "event_id": str(event["id"]), "date": str(event.get("date", ""))[:10],
            "home": home, "away": away,
            "home_goals": _int(sides["home"].get("score")), "away_goals": _int(sides["away"].get("score")),
        })
    return out


@lru_cache(maxsize=None)
def _resolver(code: str):
    from .teams import TeamResolver
    from . import league

    df = league._results().filter(pl.col("competition_code") == code)
    names = set(df["home_team"].to_list()) | set(df["away_team"].to_list())
    return TeamResolver(sorted(n for n in names if n))


def resolve_team(name: str, code: str) -> str:
    """Lake spelling of an ESPN club name; falls back to ESPN's own spelling (logged by the caller's report)."""
    return _resolver(code).resolve(name) or name


def _group(entry: dict) -> str | None:
    from .match_lineups import GROUPS
    abbreviation = (entry.get("position") or {}).get("abbreviation")
    return GROUP_LETTER.get(GROUPS.get(abbreviation))


def parse_summary(
    body: dict, code: str, meta: dict, resolve: Callable[[str, str], str] = resolve_team
) -> list[dict]:
    """One row per player who appeared (starters and used substitutes)."""
    rows: list[dict] = []
    rosters = (body or {}).get("rosters") or []
    names = {r.get("homeAway"): resolve((r.get("team") or {}).get("displayName", ""), code) for r in rosters}
    for roster in rosters:
        side = roster.get("homeAway")
        is_home = side == "home"
        other = names.get("away" if is_home else "home", "")
        for entry in roster.get("roster") or []:
            stats = {s.get("name"): s.get("value") for s in entry.get("stats") or []}
            athlete = entry.get("athlete") or {}
            appeared = _int(stats.get("appearances")) > 0 or entry.get("starter") is True
            if not appeared or not athlete.get("id"):
                continue
            rows.append({
                "espn_id": str(athlete["id"]), "player": athlete.get("displayName") or athlete.get("fullName") or "",
                "team": names.get(side, ""), "opponent": other, "competition_code": code,
                "date": meta["date"], "event_id": str(meta["event_id"]), "is_home": is_home,
                "started": entry.get("starter") is True,
                "sub_in": entry.get("starter") is not True and (bool(entry.get("subbedIn")) or _int(stats.get("subIns")) > 0),
                "position": _group(entry),
                **{col: _int(stats.get(src)) for col, src in STAT_FIELDS.items()},
                "team_goals": _int(meta["home_goals" if is_home else "away_goals"]),
                "opp_goals": _int(meta["away_goals" if is_home else "home_goals"]),
            })
    return rows


# ---------------------------------------------------------------- fetching ---

def _get(url: str, params: dict | None = None, attempts: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = httpx.get(url, params=params, timeout=25)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("unexpected ESPN payload")
            time.sleep(MIN_INTERVAL)
            return body
        except (httpx.HTTPError, ValueError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ESPN request failed after {attempts} attempts: {url} {params}: {last}")


def fetch_scoreboard(code: str, day: date, today: date | None = None) -> dict:
    """One day's scoreboard. Days older than yesterday are final and never refetched."""
    today = today or datetime.now(timezone.utc).date()
    path = RAW / code / f"sb-{day:%Y%m%d}.json"
    cached = _cache.read_json(path)
    if cached is not None and day < today - timedelta(days=1):
        return cached
    body = _get(f"{BASE}/{config.ESPN_SLUGS[code]}/scoreboard", {"dates": f"{day:%Y%m%d}", "limit": 100})
    _cache.write_json(path, body)
    return body


def fetch_summary(code: str, meta: dict) -> dict:
    """A finished match's summary, wrapped with its scoreboard meta. Cached forever."""
    path = RAW / code / f"sum-{meta['event_id']}.json"
    cached = _cache.read_json(path)
    if cached is not None and cached.get("body", {}).get("rosters"):
        return cached
    body = _get(f"{BASE}/{config.ESPN_SLUGS[code]}/summary", {"event": meta["event_id"]})
    wrapped = {"meta": meta, "body": body}
    _cache.write_json(path, wrapped)
    return wrapped


def backfill(codes: list[str] | None = None, today: date | None = None, workers: int = 4) -> dict:
    """Download every finished match of the season in progress. Idempotent and resumable."""
    today = today or datetime.now(timezone.utc).date()
    report: dict = {"scoreboards": 0, "events": 0, "summaries": 0, "errors": []}
    for code in codes or [c for c in config.LEAGUES if c in config.ESPN_SLUGS]:
        days = [season_start(code, today) + timedelta(days=i) for i in range((today - season_start(code, today)).days + 1)]

        def day_events(day: date, code: str = code) -> list[dict]:
            try:
                return completed_events(fetch_scoreboard(code, day, today))
            except RuntimeError as exc:
                report["errors"].append(str(exc))
                return []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            events = [e for batch in pool.map(day_events, days) for e in batch]
        report["scoreboards"] += len(days)
        report["events"] += len(events)

        def one(meta: dict, code: str = code) -> bool:
            try:
                fetch_summary(code, meta)
                return True
            except RuntimeError as exc:
                report["errors"].append(str(exc))
                return False

        with ThreadPoolExecutor(max_workers=workers) as pool:
            report["summaries"] += sum(pool.map(one, events))
        log.info("%s: %d days, %d finished events", code, len(days), len(events))
    return report


# ---------------------------------------------------------------- silver ---

def build_silver() -> int:
    """Re-derive `player_match_stats` from every cached summary. Returns the row count."""
    rows: list[dict] = []
    for code in config.ESPN_SLUGS:
        for path in sorted((RAW / code).glob("sum-*.json")) if (RAW / code).is_dir() else []:
            wrapped = _cache.read_json(path)
            if wrapped and not _is_exhibition(wrapped["meta"]):
                rows.extend(parse_summary(wrapped["body"], code, wrapped["meta"]))
    df = pl.DataFrame(rows, schema=SCHEMA) if rows else pl.DataFrame(schema=SCHEMA)
    df = df.unique(subset=["espn_id", "event_id"], keep="first").sort(["date", "event_id", "team", "player"])
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_path(STATS_FILE) as temporary:
        df.write_parquet(temporary, compression="zstd")
    return df.height


def leaders(code: str, limit: int = 10, frame: pl.DataFrame | None = None) -> dict:
    """Season goal and assist leaders of one competition from the box scores (for leagues with no scorer feed)."""
    df = (frame if frame is not None else load()).filter(pl.col("competition_code") == code)
    if df.height == 0:
        return {"scorers": [], "assists": []}
    totals = (df.group_by(["espn_id", "player", "team"])
                .agg(pl.col("goals").sum(), pl.col("assists").sum(), pl.col("event_id").n_unique().alias("played")))

    def board(column: str) -> list[dict]:
        top = totals.filter(pl.col(column) > 0).sort([column, "played", "player"], descending=[True, False, False]).head(limit)
        return [{"player": r["player"], "team": r["team"], "goals": r["goals"], "assists": r["assists"], "played": r["played"],
                 "crest": None, "player_id": f"espn-{r['espn_id']}"} for r in top.to_dicts()]

    return {"scorers": board("goals"), "assists": board("assists")}


def load() -> pl.DataFrame:
    """The silver table, or an empty frame with the right schema if the backfill has not run."""
    if not STATS_FILE.exists():
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(STATS_FILE)
