"""Build the unified fixtures table.

Combines the rolling football-data.co.uk fixture file (kick-off times + odds,
next few days) with openfootball full-season schedules (whole season, no odds),
reconciling club names onto the football-data spelling so fixtures join cleanly
to the results lake and to standings.

Each row is stamped with a status derived from the clock:

    finished   a score is known
    live       kick-off has passed and the match is inside its expected window
    scheduled  kick-off is still ahead

`live` is honest about what it is: derived from real kick-off times, not from an
in-play feed. It answers "is this match under way right now", which is what the
live counter needs, without pretending to know the current score.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import polars as pl

from .. import config
from ..storage import atomic_path
from ..sources import espn_fixtures as espn_fx
from ..sources import fixtures as fx_src
from ..sources import football_data_org as fdo
from ..teams import TeamResolver

log = logging.getLogger(__name__)

SILVER_FIXTURES = config.SILVER / "fixtures"

SCHEMA = {
    "competition_code": pl.Utf8, "date": pl.Utf8, "time": pl.Utf8,
    "kickoff_utc": pl.Utf8,
    "home_team": pl.Utf8, "away_team": pl.Utf8, "referee": pl.Utf8,
    "round": pl.Utf8,
    "home_goals": pl.Int32, "away_goals": pl.Int32,
    "odds_home": pl.Float64, "odds_draw": pl.Float64, "odds_away": pl.Float64,
    "source": pl.Utf8, "fixture_key": pl.Utf8, "status": pl.Utf8,
}


def _resolvers() -> dict[str, TeamResolver]:
    res = pl.read_parquet(config.SILVER / "results" / "results.parquet")
    out: dict[str, TeamResolver] = {}
    for code in res["competition_code"].unique().to_list():
        sub = res.filter(pl.col("competition_code") == code)
        names = sorted(set(sub["home_team"].to_list()) | set(sub["away_team"].to_list()))
        out[code] = TeamResolver(names)
    return out


def _kickoff(date_s: str, time_s: str | None) -> str | None:
    if not date_s:
        return None
    hhmm = (time_s or "00:00")[:5]
    try:
        naive = datetime.strptime(f"{date_s} {hhmm}", "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            naive = datetime.strptime(date_s, "%Y-%m-%d")
        except ValueError:
            return None
    # Source times are local to the competition; treated as UTC because the
    # files carry no zone. Good enough for "is it on now", and never presented
    # as an exact local kick-off.
    return naive.replace(tzinfo=timezone.utc).isoformat()


def _collapse_same_fixture(df: pl.DataFrame) -> pl.DataFrame:
    """Merge one match listed on adjacent dates by two sources.

    `fixture_key` includes the date, so Barcelona v Getafe appeared twice:
    openfootball dates it 11 October with no kick-off time, football-data.org
    has it 10 October at 16:30. Same match, two rows, and the timeless one
    rendered as midnight — 8:00 PM the previous evening once converted.

    Rows for the same competition and pairing within a day of each other are
    collapsed, keeping the one that carries an actual kick-off time.
    """
    if df.height == 0:
        return df

    ranked = df.with_columns(
        # A real kick-off beats a date-only entry; among those, the row with
        # odds and a referee is the better record.
        pl.when(pl.col("time").is_null() | (pl.col("time") == "00:00"))
          .then(1).otherwise(0).alias("_timeless"),
        pl.col("odds_home").is_null().cast(pl.Int8).alias("_no_odds"),
    ).sort(["_timeless", "_no_odds", "date"], maintain_order=True)

    keep: list[bool] = []
    seen: dict[tuple[str, str, str], list[str]] = {}
    for row in ranked.iter_rows(named=True):
        pairing = (row["competition_code"], row["home_team"], row["away_team"])
        dates = seen.setdefault(pairing, [])
        near = any(abs(_day_gap(row["date"], d)) <= 1 for d in dates)
        keep.append(not near)
        if not near:
            dates.append(row["date"])

    return (
        ranked.filter(pl.Series(keep))
              .drop("_timeless", "_no_odds")
    )


def _day_gap(a: str, b: str) -> int:
    try:
        left = datetime.strptime(a[:10], "%Y-%m-%d")
        right = datetime.strptime(b[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return 99
    return (left - right).days


def _status(kickoff_iso: str | None, has_score: bool, now: datetime) -> str:
    if has_score:
        return "finished"
    if not kickoff_iso:
        return "scheduled"
    ko = datetime.fromisoformat(kickoff_iso)
    if ko > now:
        return "scheduled"
    if now <= ko + timedelta(minutes=config.MATCH_DURATION_MINUTES):
        return "live"
    return "finished_unknown"


def build(now: datetime | None = None) -> dict[str, int]:
    config.ensure_dirs()
    SILVER_FIXTURES.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(timezone.utc)

    resolvers = _resolvers()
    rows: list[dict] = []

    def add(rec: dict, canonical_needed: bool) -> None:
        code = rec["competition_code"]
        home, away = rec["home_team"], rec["away_team"]
        if canonical_needed:
            r = resolvers.get(code)
            if r is None:
                return
            home_c, away_c = r.resolve(home), r.resolve(away)
            if not home_c or not away_c:
                # Unresolvable names are dropped, never guessed: a wrong merge
                # corrupts a league table in a way that is hard to spot.
                return
            home, away = home_c, away_c
        ko = _kickoff(rec["date"], rec.get("time"))
        has_score = rec.get("home_goals") is not None and rec.get("away_goals") is not None
        rows.append({
            "competition_code": code,
            "date": rec["date"],
            "time": rec.get("time"),
            "kickoff_utc": ko,
            "home_team": home,
            "away_team": away,
            "referee": rec.get("referee"),
            "round": rec.get("round"),
            "home_goals": rec.get("home_goals"),
            "away_goals": rec.get("away_goals"),
            "odds_home": rec.get("odds_home"),
            "odds_draw": rec.get("odds_draw"),
            "odds_away": rec.get("odds_away"),
            "source": rec.get("source"),
            "fixture_key": f"{code}|{rec['date']}|{home}|{away}",
            "status": _status(ko, has_score, now),
        })

    # Season schedules first, then the rolling file, so the rolling rows (which
    # carry odds and a referee) win on duplicate keys.
    for rec in fx_src.read_season_schedules():
        add(rec, canonical_needed=True)
    for rec in fx_src.read_football_data_fixtures():
        add(rec, canonical_needed=False)

    # football-data.org last: it is the only source of Champions League
    # fixtures, since both sources above publish domestic divisions only. Its
    # names already match the lake for UCL, so they pass through uncanonicalised
    # there; domestic rows still go through the resolver.
    for code in fdo.COMPETITIONS:
        for rec in fdo.read_fixtures(code):
            add(rec, canonical_needed=(code != "UCL"))

    # Competitions no keyed provider serves (MLS) come from ESPN's scoreboard.
    for code in espn_fx.ESPN_ONLY:
        for rec in espn_fx.read(code):
            add(rec, canonical_needed=True)

    if not rows:
        return {"fixtures": 0}

    df = pl.DataFrame(rows, schema=SCHEMA).unique(subset=["fixture_key"], keep="last")
    df = _collapse_same_fixture(df)
    df = df.sort(["date", "time", "competition_code"])
    with atomic_path(SILVER_FIXTURES / "fixtures.parquet") as temporary:
        df.write_parquet(temporary, compression="zstd")

    counts = {"fixtures": df.height}
    for st in ("scheduled", "live", "finished", "finished_unknown"):
        counts[st] = int((df["status"] == st).sum())
    counts["with_odds"] = int(df["odds_home"].is_not_null().sum())
    return counts
