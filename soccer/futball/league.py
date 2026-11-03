"""League-level query layer — standings, matches, fixtures and live status.

This module reads the RESULTS lake (football-data.co.uk + openfootball), which
holds club competitions. It is deliberately separate from `marts.py`, which
serves StatsBomb event data for the 3D match centre and covers one international
tournament. Mixing the two is what previously put national teams in a league
table.

Live status is computed at QUERY time from real kick-off times, never stored,
because a status baked in at build time is wrong the moment the clock moves.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta, timezone

import polars as pl

from . import conferences, config
from .cache import file_cache

log = logging.getLogger(__name__)

_lock = threading.Lock()

RESULTS = config.SILVER / "results" / "results.parquet"
FIXTURES = config.SILVER / "fixtures" / "fixtures.parquet"


# --------------------------------------------------------------------------
# season handling — three sources spell seasons three different ways
# --------------------------------------------------------------------------

def season_start_year(label: str | None) -> int | None:
    """'2026/27' -> 2026 · '2026' -> 2026 · '2025-26' -> 2025."""
    if not label:
        return None
    text = str(label).strip()
    for sep in ("/", "-"):
        if sep in text:
            head = text.split(sep)[0]
            if head.isdigit():
                return int(head) if len(head) == 4 else 2000 + int(head)
    return int(text) if text.isdigit() and len(text) == 4 else None


def current_season_year(code: str, today: date | None = None) -> int:
    """Which season is 'now' for this competition."""
    today = today or date.today()
    meta = config.LEAGUES.get(code, {})
    # MLS runs inside a calendar year; European competitions straddle two.
    if code == "MLS":
        return today.year
    return today.year if today.month >= 7 else today.year - 1


@file_cache(lambda: RESULTS)
def _results() -> pl.DataFrame:
    config.require_volume()
    df = pl.read_parquet(RESULTS)
    years = [season_start_year(s) for s in df["season"].to_list()]
    return df.with_columns(pl.Series("season_year", years, dtype=pl.Int32))


@file_cache(lambda: FIXTURES)
def _fixtures() -> pl.DataFrame:
    config.require_volume()
    if not FIXTURES.exists():
        return pl.DataFrame(schema={"competition_code": pl.Utf8, "date": pl.Utf8})
    return pl.read_parquet(FIXTURES)


def refresh_caches() -> None:
    with _lock:
        _results.cache_clear()
        _fixtures.cache_clear()


# --------------------------------------------------------------------------
# competitions
# --------------------------------------------------------------------------

def leagues() -> list[dict]:
    """The seven competitions, with what data actually exists for each."""
    df = _results()
    out: list[dict] = []
    for code, meta in config.LEAGUES.items():
        sub = df.filter(pl.col("competition_code") == code)
        cur_year = current_season_year(code)
        cur = sub.filter(pl.col("season_year") == cur_year)
        seasons_available = sorted(
            {y for y in sub["season_year"].to_list() if y is not None}, reverse=True
        )
        latest_with_data = seasons_available[0] if seasons_available else None
        label = None
        if cur.height:
            label = cur["season"].to_list()[0]
        elif latest_with_data is not None:
            label = sub.filter(pl.col("season_year") == latest_with_data)["season"].to_list()[0]
        out.append({
            "code": code,
            "name": meta["name"],
            "country": meta["country"],
            "kind": meta["kind"],
            # Official competition badge, so the sidebar shows real logos
            # rather than a generic trophy glyph for all three.
            "emblem": config.COMPETITION_EMBLEMS.get(code),
            "total_matches": sub.height,
            "current_season_year": cur_year,
            "current_season_matches": cur.height,
            "displayed_season_year": cur_year if cur.height else latest_with_data,
            "displayed_season": label,
            "seasons": seasons_available[:12],
            "last_match": sub["date"].max() if sub.height else None,
        })
    return out


def seasons(code: str) -> list[dict]:
    df = _results().filter(pl.col("competition_code") == code)
    if df.height == 0:
        return []
    return (
        df.group_by(["season_year", "season"])
        .agg(pl.len().alias("matches"), pl.col("date").max().alias("last_match"))
        .sort("season_year", descending=True)
        .to_dicts()
    )


def _resolve_season(code: str, season_year: int | None) -> tuple[int | None, bool]:
    """Returns (season_year_used, is_fallback)."""
    df = _results().filter(pl.col("competition_code") == code)
    if df.height == 0:
        return None, False
    available = sorted({y for y in df["season_year"].to_list() if y is not None}, reverse=True)
    if season_year is not None:
        return (season_year, False) if season_year in available else (
            (available[0], True) if available else (None, False)
        )
    cur = current_season_year(code)
    if cur in available and df.filter(pl.col("season_year") == cur).height > 0:
        return cur, False
    return (available[0], True) if available else (None, False)


# --------------------------------------------------------------------------
# standings
# --------------------------------------------------------------------------

def standings(code: str, season_year: int | None = None) -> dict:
    """League table for one competition and season.

    Returns the table plus the context needed to be honest about it: which
    season is shown, whether that is a fallback because the requested season has
    not started, and how many matches it is based on. A table built from two
    matchdays is not wrong, but the reader must be told.
    """
    used, fallback = _resolve_season(code, season_year)
    meta = config.LEAGUES.get(code, {"name": code, "country": ""})
    if used is None:
        return {
            "competition_code": code, "competition": meta["name"],
            "season_year": None, "season": None, "is_fallback": False,
            "matches_played": 0, "rows": [],
            "note": f"No results are available for {meta['name']} yet.",
        }

    df = _results().filter(
        (pl.col("competition_code") == code) & (pl.col("season_year") == used)
    )
    label = df["season"].to_list()[0] if df.height else str(used)

    home = df.select(
        pl.col("home_team").alias("team"), pl.col("home_goals").alias("gf"),
        pl.col("away_goals").alias("ga"), pl.col("date"),
    )
    away = df.select(
        pl.col("away_team").alias("team"), pl.col("away_goals").alias("gf"),
        pl.col("home_goals").alias("ga"), pl.col("date"),
    )
    sided = pl.concat([home, away]).drop_nulls(["gf", "ga"])

    if sided.height == 0:
        return {
            "competition_code": code, "competition": meta["name"],
            "season_year": used, "season": label, "is_fallback": fallback,
            "matches_played": 0, "rows": [],
            "note": f"{meta['name']} {label} has no completed matches recorded yet.",
        }

    table = (
        sided.group_by("team")
        .agg(
            pl.len().alias("played"),
            (pl.col("gf") > pl.col("ga")).sum().alias("won"),
            (pl.col("gf") == pl.col("ga")).sum().alias("drawn"),
            (pl.col("gf") < pl.col("ga")).sum().alias("lost"),
            pl.col("gf").sum().alias("goals_for"),
            pl.col("ga").sum().alias("goals_against"),
        )
        .with_columns(
            (pl.col("goals_for") - pl.col("goals_against")).alias("goal_difference"),
            (pl.col("won") * 3 + pl.col("drawn")).alias("points"),
        )
        .sort(["points", "goal_difference", "goals_for", "team"],
              descending=[True, True, True, False])
    )

    form_rows = (
        sided.with_columns(
            pl.when(pl.col("gf") > pl.col("ga")).then(pl.lit("W"))
            .when(pl.col("gf") == pl.col("ga")).then(pl.lit("D"))
            .otherwise(pl.lit("L")).alias("r")
        )
        .sort(["team", "date"])
        .group_by("team", maintain_order=True)
        .agg(pl.col("r"))
        .to_dicts()
    )
    form = {r["team"]: r["r"][-5:] for r in form_rows}

    upcoming = _next_opponents(code)
    membership = conferences.conferences(code)  # MLS only; {} elsewhere or if ESPN is unavailable
    ordered = table.to_dicts()
    if membership:
        ranked = conferences.rank_in_conferences(ordered, membership, lambda pos: config.qualification_band(code, pos))
    else:
        # A conference league with no membership shows one combined table and NO band: the playoff line is per conference.
        no_band = code in conferences.CONFERENCE_LEAGUES
        ranked = []
        for i, r in enumerate(ordered, 1):
            band = None if no_band else config.qualification_band(code, i)
            ranked.append({**r, "conference": None, "position": i,
                           "qualification": band[0] if band else None, "qualification_tone": band[1] if band else None})
    rows = [{**r, "form": form.get(r["team"], []), "next": upcoming.get(r["team"])} for r in ranked]
    note = None
    if fallback:
        note = (
            f"The {current_season_year(code)}/{str(current_season_year(code)+1)[2:]} "
            f"season has no completed matches recorded yet, so {label} is shown."
        )
    elif df.height < 20:
        note = f"{label} has just started — this table is based on {df.height} matches."

    return {
        "competition_code": code, "competition": meta["name"],
        "season_year": used, "season": label, "is_fallback": fallback,
        "matches_played": df.height, "rows": rows, "note": note,
    }


# --------------------------------------------------------------------------
# matches & fixtures
# --------------------------------------------------------------------------

def matches(code: str, season_year: int | None = None, limit: int = 200) -> dict:
    used, fallback = _resolve_season(code, season_year)
    meta = config.LEAGUES.get(code, {"name": code})
    if used is None:
        return {"competition_code": code, "season_year": None, "season": None, "rows": []}
    df = (
        _results()
        .filter((pl.col("competition_code") == code) & (pl.col("season_year") == used))
        .sort("date", descending=True)
        .head(limit)
    )
    label = df["season"].to_list()[0] if df.height else str(used)
    return {
        "competition_code": code, "competition": meta["name"],
        "season_year": used, "season": label, "is_fallback": fallback,
        "rows": df.select([
            "date", "time", "home_team", "away_team", "home_goals", "away_goals",
            "result", "total_goals", "odds_home", "odds_draw", "odds_away",
            # match detail — present for the domestic divisions, null for UCL
            "ht_home_goals", "ht_away_goals",
            "home_shots", "away_shots", "home_sot", "away_sot",
            "home_corners", "away_corners", "home_fouls", "away_fouls",
            "home_yellow", "away_yellow", "home_red", "away_red",
            "home_xg", "away_xg",
        ]).to_dicts(),
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _with_live_status(df: pl.DataFrame) -> pl.DataFrame:
    """Recompute status against the clock, right now."""
    now = _now()
    horizon = now - timedelta(minutes=config.MATCH_DURATION_MINUTES)
    return df.with_columns(
        pl.when(pl.col("home_goals").is_not_null() & pl.col("away_goals").is_not_null())
        .then(pl.lit("finished"))
        .when(pl.col("kickoff_utc").is_null())
        .then(pl.lit("scheduled"))
        .when(pl.col("kickoff_utc") > pl.lit(now.isoformat()))
        .then(pl.lit("scheduled"))
        .when(pl.col("kickoff_utc") > pl.lit(horizon.isoformat()))
        .then(pl.lit("live"))
        .otherwise(pl.lit("finished_unknown"))
        .alias("live_status")
    )


def fixtures(code: str | None = None, days: int = 10, limit: int = 200) -> list[dict]:
    """Upcoming and in-progress fixtures, nearest first.

    The rolling fixture file covers ~20 divisions; only the competitions this
    dashboard displays are returned, so counts and lists never include leagues
    the user cannot navigate to.
    """
    df = _fixtures()
    if df.height == 0:
        return []
    df = df.filter(pl.col("competition_code").is_in(list(config.LEAGUES)))
    today = date.today().isoformat()
    until = (date.today() + timedelta(days=days)).isoformat()
    df = df.filter((pl.col("date") >= today) & (pl.col("date") <= until))
    if code:
        df = df.filter(pl.col("competition_code") == code)
    df = _with_live_status(df).filter(pl.col("live_status").is_in(["scheduled", "live"]))
    df = df.sort(["date", "time"]).head(limit)
    return df.select([
        "competition_code", "date", "time", "kickoff_utc", "home_team", "away_team",
        "referee", "odds_home", "odds_draw", "odds_away", "live_status", "fixture_key",
    ]).to_dicts()


def live_now(code: str | None = None) -> list[dict]:
    from .live.espn import counts
    return [r for r in counts()['matches'] if code is None or r['competition_code'] == code]


# --------------------------------------------------------------------------
# live scores
#
# The feed is ESPN (`futball/live/espn.py`), which publishes a real status per
# match — so nothing here infers "in play" from a kick-off time plus an assumed
# duration, and it needs no key.
#
# Provider names are long display forms ("Rayo Vallecano") while the lake uses
# football-data.co.uk's short ones ("Vallecano"), so the two have to be
# reconciled before a score can be attached to a table row.
# --------------------------------------------------------------------------


@file_cache(lambda: config.SILVER / "results" / "results.parquet", maxsize=8)
def _lake_name_by_club(competition_code: str) -> dict[str, str]:
    """Club-directory canonical name -> the lake's spelling, within one competition.

    Scoped per competition because the lake is *not* consistent across them:
    football-data.co.uk writes "Valencia" for La Liga while openfootball writes
    "Valencia CF" for the Champions League, and both resolve to the same club.
    A single global index picked between them by set-iteration order, which
    attached a La Liga live score to a row that did not exist.
    """
    from . import clubs

    df = _results().filter(pl.col("competition_code") == competition_code)

    def tally(frame: pl.DataFrame) -> dict[str, int]:
        counts: dict[str, int] = {}
        for column in ("home_team", "away_team"):
            for name, n in frame[column].value_counts().iter_rows():
                if name:
                    counts[name] = counts.get(name, 0) + n
        return counts

    # Weight the last two seasons only. A club's spelling changes between
    # sources over the years — openfootball wrote "Real Madrid" in 2018 and
    # "Real Madrid CF" now — and an all-time count elects the historic form,
    # which is not the one the rest of the app is keyed on today. `serve.teams`
    # already reads the same window, so both agree on one spelling per club.
    seasons = sorted(df["season"].unique().to_list())[-2:]
    recent = tally(df.filter(pl.col("season").is_in(seasons)))
    historic = tally(df)

    # One club can carry several spellings inside a single competition —
    # openfootball writes "Club Atlético de Madrid" in some seasons and
    # "Atlético Madrid" in others. Take the spelling the most rows use, with the
    # name as a tie-break: picking arbitrarily meant a live score or a forecast
    # attached itself to a spelling the rest of the app had never heard of.
    def elect(counts: dict[str, int], into: dict[str, tuple[int, str]]) -> None:
        for name, count in counts.items():
            entry = clubs.find(name)
            if not entry:
                continue
            current = into.get(entry["name"])
            if current is None or (count, name) > current:
                into[entry["name"]] = (count, name)

    best: dict[str, tuple[int, str]] = {}
    elect(recent, best)

    # Only now fall back to the full history, and only for clubs the recent
    # window never saw. Mixing the two tallies together let a historic spelling
    # win on sheer volume — "Real Madrid" across six old seasons outvoting
    # "Real Madrid CF" in the season actually being played.
    older: dict[str, tuple[int, str]] = {}
    elect(historic, older)
    for club, spelling in older.items():
        best.setdefault(club, spelling)

    return {club: spelling for club, (_n, spelling) in best.items()}


def _to_lake_name(provider_name: str | None, competition_code: str) -> str | None:
    from . import clubs

    if not provider_name:
        return None
    entry = clubs.find(provider_name)
    if entry is None:
        return None
    return _lake_name_by_club(competition_code).get(entry["name"])


def live_scores(code: str | None = None) -> dict[str, dict]:
    return {f"{r['competition_code']}|{r['home_team']}|{r['away_team']}":
            {**r, 'status': 'PAUSED' if r['provider_status'] == 'STATUS_HALFTIME' else 'IN_PLAY'}
            for r in live_now(code) if r['home_goals'] is not None and r['away_goals'] is not None}


def live_counts() -> dict:
    from .live.espn import counts
    return counts()


# --------------------------------------------------------------------------
# match context — used by the live view, which has no in-play feed
# --------------------------------------------------------------------------

def head_to_head(code: str, home: str, away: str, limit: int = 6) -> list[dict]:
    """Previous meetings between two clubs, most recent first."""
    df = _results().filter(
        (pl.col("competition_code") == code)
        & (
            ((pl.col("home_team") == home) & (pl.col("away_team") == away))
            | ((pl.col("home_team") == away) & (pl.col("away_team") == home))
        )
    ).sort("date", descending=True).head(limit)
    return df.select([
        "date", "season", "home_team", "away_team", "home_goals", "away_goals", "result",
    ]).to_dicts()


def team_form(code: str, team: str, limit: int = 6) -> list[dict]:
    """A club's most recent results in this competition, most recent first."""
    df = _results().filter(
        (pl.col("competition_code") == code)
        & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
    ).sort("date", descending=True).head(limit)
    rows = []
    for r in df.to_dicts():
        at_home = r["home_team"] == team
        gf = r["home_goals"] if at_home else r["away_goals"]
        ga = r["away_goals"] if at_home else r["home_goals"]
        rows.append({
            "date": r["date"],
            "opponent": r["away_team"] if at_home else r["home_team"],
            "at_home": at_home,
            "goals_for": gf, "goals_against": ga,
            "outcome": "W" if gf > ga else "D" if gf == ga else "L",
        })
    return rows


# ---------------------------------------------------------------------------
# clean sheets
#
# Neither football-data.org nor API-Football publishes a clean-sheet
# leaderboard, and Wikipedia carries goalkeeper clean sheets for the Premier
# League only. But a clean sheet is just "conceded zero", and the results lake
# already holds every scoreline — so this is computed exactly, for every
# competition and season, at no API cost.
#
# It is a TEAM board, not a goalkeeper one. Attributing a clean sheet to a
# keeper needs lineups, which no free source provides for these competitions,
# and guessing the regular starter would silently misattribute the matches he
# did not play.
# ---------------------------------------------------------------------------

def _next_opponents(code: str) -> dict[str, dict]:
    """Each club's next scheduled fixture in this competition.

    The table's "Next" column is the one part of it that looks forward, so it
    reads from the fixture layer rather than the results lake. A club with no
    scheduled fixture simply has no entry.
    """
    try:
        upcoming = fixtures(code=code, days=45, limit=400)
    except Exception:  # noqa: BLE001 - a missing fixture layer must not break the table
        return {}

    out: dict[str, dict] = {}
    for f in sorted(upcoming, key=lambda f: (f.get("date") or "", f.get("time") or "")):
        for team, opponent, at_home in (
            (f.get("home_team"), f.get("away_team"), True),
            (f.get("away_team"), f.get("home_team"), False),
        ):
            if team and opponent and team not in out:
                out[team] = {
                    "opponent": opponent,
                    "at_home": at_home,
                    "date": f.get("date"),
                    "time": f.get("time"),
                }
    return out


def clean_sheets(code: str, season_year: int | None = None, limit: int = 20) -> dict:
    """Clean sheets per team, computed from match results."""
    used, fallback = _resolve_season(code, season_year)
    meta = config.LEAGUES.get(code, {})
    if used is None:
        return {
            "competition_code": code, "competition": meta.get("name", code),
            "season_year": None, "season": None, "is_fallback": False,
            "rows": [], "note": "No results recorded for this competition.",
        }

    df = _results().filter(
        (pl.col("competition_code") == code) & (pl.col("season_year") == used)
    )
    label = df["season"][0] if df.height else None

    home = df.select(
        team=pl.col("home_team"),
        conceded=pl.col("away_goals"),
        scored=pl.col("home_goals"),
    )
    away = df.select(
        team=pl.col("away_team"),
        conceded=pl.col("home_goals"),
        scored=pl.col("away_goals"),
    )
    both = pl.concat([home, away])

    table = (
        both.group_by("team")
        .agg(
            played=pl.len(),
            clean_sheets=(pl.col("conceded") == 0).sum(),
            goals_conceded=pl.col("conceded").sum(),
            failed_to_score=(pl.col("scored") == 0).sum(),
        )
        .with_columns(
            clean_sheet_rate=(pl.col("clean_sheets") / pl.col("played")).round(3)
        )
        .sort(["clean_sheets", "goals_conceded"], descending=[True, False])
        .head(limit)
    )

    rows = [
        {**r, "position": i} for i, r in enumerate(table.to_dicts(), 1)
    ]
    note = None
    if fallback:
        note = (
            f"The {current_season_year(code)}/{str(current_season_year(code) + 1)[2:]} "
            f"season has no completed matches recorded yet, so {label} is shown."
        )
    elif df.height < 20:
        note = f"{label} has just started — based on {df.height} matches."

    return {
        "competition_code": code, "competition": meta.get("name", code),
        "season_year": used, "season": label, "is_fallback": fallback,
        "matches_played": df.height, "rows": rows, "note": note,
    }


# ---------------------------------------------------------------------------
# season honours
#
# A cup is decided by its final; a league by the top of its table. Both are
# derivable here, but only if the final can be *identified* — and the obvious
# shortcut of taking the last match by date is wrong. Champions League 2025-26
# ends on a drawn semi-final, so a date heuristic would print a final nobody
# played.
# ---------------------------------------------------------------------------

def _is_final(stage: str | None) -> bool:
    return bool(stage) and stage.strip().lower().endswith("final")


def season_honours(code: str) -> list[dict]:
    """Winner and runner-up per season, newest first.

    `decided_by` says how each was resolved, so the page can be honest about a
    gap rather than showing an empty card or inventing a winner.
    """
    meta = config.LEAGUES.get(code, {})
    if not meta:
        return []

    df = _results().filter(pl.col("competition_code") == code)
    if df.height == 0:
        return []

    is_cup = meta.get("kind") == "cup"
    # A season is live if it is *this* season, not if its last played match is
    # in the past. Using the latest result would call 2026/27 complete every
    # week between rounds.
    live_year = current_season_year(code)
    out: list[dict] = []

    for season in sorted(df["season"].unique().to_list(), reverse=True):
        sub = df.filter(pl.col("season") == season)
        year = season_start_year(season)
        entry = {
            "season": season, "season_year": year,
            "winner": None, "runner_up": None,
            "decided_by": None, "note": None,
            "matches": sub.height,
        }

        if is_cup:
            finals = sub.filter(
                pl.col("stage").is_not_null()
                & pl.col("stage").map_elements(_is_final, return_dtype=pl.Boolean)
            ).sort("date")
            if finals.height == 0:
                still_running = year == live_year
                entry["decided_by"] = "in_progress" if still_running else "unavailable"
                entry["note"] = (
                    "Season in progress." if still_running
                    else "The source has not published a final for this season."
                )
            else:
                f = finals.tail(1).to_dicts()[0]
                if f["home_goals"] == f["away_goals"]:
                    # Shoot-outs are not in the scoreline, so the winner cannot
                    # be read off a drawn final.
                    entry["decided_by"] = "undecided"
                    entry["note"] = (
                        f"Final drawn {f['home_goals']}-{f['away_goals']}; "
                        f"the shoot-out result is not in this data."
                    )
                else:
                    home_won = f["home_goals"] > f["away_goals"]
                    entry["winner"] = f["home_team"] if home_won else f["away_team"]
                    entry["runner_up"] = f["away_team"] if home_won else f["home_team"]
                    entry["decided_by"] = "final"
                    entry["note"] = (
                        f"{f['home_team']} {f['home_goals']}-{f['away_goals']} {f['away_team']}"
                    )
        else:
            table = standings(code, year)
            rows = table.get("rows") or []
            if len(rows) < 2:
                entry["decided_by"] = "in_progress"
                entry["note"] = "Not enough matches recorded yet."
            else:
                complete = year != live_year
                entry["winner"] = rows[0]["team"]
                entry["runner_up"] = rows[1]["team"]
                entry["decided_by"] = "table" if complete else "in_progress"
                if not complete:
                    entry["note"] = "Season in progress — current standings."

        out.append(entry)
    return out
