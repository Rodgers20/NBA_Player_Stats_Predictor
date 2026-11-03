"""Normalise match results from every source into one schema.

Two shapes are unified here:

* football-data.co.uk **division** files — one CSV per season per division, with
  goals, half-time score, shots, shots on target, corners, fouls, cards, closing
  odds and (2026/27 onward) match xG.
* football-data.co.uk **extra** files — one multi-season CSV per country, goals
  and odds only. MLS lives here.

The union schema is the superset; a source that lacks a column contributes null
rather than a fabricated zero, because "no shots recorded" and "zero shots" are
very different facts and a model must be able to tell them apart.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import polars as pl

from .. import config
from ..storage import atomic_path
from ..sources import football_data as fd
from ..sources import football_data_org as fdo
from ..sources import openfootball as of
from ..teams import TeamResolver

log = logging.getLogger(__name__)

SILVER_RESULTS = config.SILVER / "results"


def _f(v: Any) -> float | None:
    if v in (None, "", "NA"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    f = _f(v)
    return int(f) if f is not None else None


def _date(v: str | None) -> str | None:
    """football-data uses dd/mm/yy and dd/mm/yyyy interchangeably."""
    if not v:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(v.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _season_label(code: str) -> str:
    """2627 -> 2026/27"""
    start = int(code[:2])
    century = 2000 if start < 90 else 1900
    return f"{century + start}/{code[2:]}"


ROW_SCHEMA = {
    "match_key": pl.Utf8, "competition_code": pl.Utf8, "competition": pl.Utf8,
    "country": pl.Utf8, "season": pl.Utf8, "date": pl.Utf8, "time": pl.Utf8,
    "home_team": pl.Utf8, "away_team": pl.Utf8,
    "home_goals": pl.Int32, "away_goals": pl.Int32, "result": pl.Utf8,
    "ht_home_goals": pl.Int32, "ht_away_goals": pl.Int32,
    "home_shots": pl.Int32, "away_shots": pl.Int32,
    "home_sot": pl.Int32, "away_sot": pl.Int32,
    "home_corners": pl.Int32, "away_corners": pl.Int32,
    "home_fouls": pl.Int32, "away_fouls": pl.Int32,
    "home_yellow": pl.Int32, "away_yellow": pl.Int32,
    "home_red": pl.Int32, "away_red": pl.Int32,
    "home_xg": pl.Float64, "away_xg": pl.Float64,
    "odds_home": pl.Float64, "odds_draw": pl.Float64, "odds_away": pl.Float64,
    "odds_over25": pl.Float64, "odds_under25": pl.Float64,
    "total_goals": pl.Int32, "both_scored": pl.Boolean, "over25": pl.Boolean,
    # Which feed this row came from. Drives dedupe precedence when two sources
    # carry the same match; see SOURCE_RANK.
    "source": pl.Utf8,
    # Competition round, where the source names one. Only openfootball's cup
    # files do, and it is what distinguishes a final from a group game — the
    # last match by date is not the final, as Champions League 2025-26 shows.
    "stage": pl.Utf8,
}


def _row(**kw: Any) -> dict:
    base = {k: None for k in ROW_SCHEMA}
    base.update(kw)
    return base


def _finish(row: dict) -> dict | None:
    hg, ag = row["home_goals"], row["away_goals"]
    if hg is None or ag is None or not row["date"] or not row["home_team"]:
        return None
    row["result"] = "H" if hg > ag else "A" if ag > hg else "D"
    row["total_goals"] = hg + ag
    row["both_scored"] = hg > 0 and ag > 0
    row["over25"] = (hg + ag) > 2.5
    row["match_key"] = (
        f"{row['competition_code']}|{row['season']}|{row['date']}|"
        f"{row['home_team']}|{row['away_team']}"
    )
    return row


def parse_division(season_code: str, div: str) -> list[dict]:
    name, country = config.FD_DIVISIONS.get(div, (div, ""))
    out = []
    for r in fd.read_division(season_code, div):
        row = _row(
            source="football-data.co.uk",
            competition_code=div, competition=name, country=country,
            season=_season_label(season_code),
            date=_date(r.get("Date")), time=(r.get("Time") or None),
            home_team=(r.get("HomeTeam") or "").strip() or None,
            away_team=(r.get("AwayTeam") or "").strip() or None,
            home_goals=_i(r.get("FTHG")), away_goals=_i(r.get("FTAG")),
            ht_home_goals=_i(r.get("HTHG")), ht_away_goals=_i(r.get("HTAG")),
            home_shots=_i(r.get("HS")), away_shots=_i(r.get("AS")),
            home_sot=_i(r.get("HST")), away_sot=_i(r.get("AST")),
            home_corners=_i(r.get("HC")), away_corners=_i(r.get("AC")),
            home_fouls=_i(r.get("HF")), away_fouls=_i(r.get("AF")),
            home_yellow=_i(r.get("HY")), away_yellow=_i(r.get("AY")),
            home_red=_i(r.get("HR")), away_red=_i(r.get("AR")),
            home_xg=_f(r.get("HxG")), away_xg=_f(r.get("AxG")),
            odds_home=_f(r.get("AvgH") or r.get("B365H")),
            odds_draw=_f(r.get("AvgD") or r.get("B365D")),
            odds_away=_f(r.get("AvgA") or r.get("B365A")),
            odds_over25=_f(r.get("B365>2.5")),
            odds_under25=_f(r.get("B365<2.5")),
        )
        done = _finish(row)
        if done:
            out.append(done)
    return out


def parse_extra(code: str) -> list[dict]:
    _url, name, country = config.FD_EXTRA[code]
    out = []
    for r in fd.read_extra(code):
        row = _row(
            source="football-data.co.uk",
            competition_code=code, competition=name, country=country,
            season=(r.get("Season") or "").strip() or None,
            date=_date(r.get("Date")), time=(r.get("Time") or None),
            home_team=(r.get("Home") or "").strip() or None,
            away_team=(r.get("Away") or "").strip() or None,
            home_goals=_i(r.get("HG")), away_goals=_i(r.get("AG")),
            odds_home=_f(r.get("AvgCH") or r.get("PSCH")),
            odds_draw=_f(r.get("AvgCD") or r.get("PSCD")),
            odds_away=_f(r.get("AvgCA") or r.get("PSCA")),
        )
        done = _finish(row)
        if done:
            out.append(done)
    return out


def parse_champions_league() -> list[dict]:
    """UEFA Champions League from openfootball.

    No shots or odds are available for this source, so those columns stay null
    rather than zero — the prediction layer then correctly falls back to its
    form-only models for European fixtures.
    """
    out = []
    for r in of.read_champions_league():
        row = _row(
            source="openfootball",
            competition_code="UCL", competition="UEFA Champions League",
            country="Europe", season=r["season"], date=r["date"], time=r.get("time"),
            stage=r.get("stage"),
            home_team=r["home_team"], away_team=r["away_team"],
            home_goals=r["home_goals"], away_goals=r["away_goals"],
            ht_home_goals=r.get("ht_home_goals"), ht_away_goals=r.get("ht_away_goals"),
        )
        done = _finish(row)
        if done:
            out.append(done)
    return out


# ---------------------------------------------------------------------------
# football-data.org — the gap-filling source
#
# It supplies goals only, for matches football-data.co.uk has not published
# yet, plus the Champions League season openfootball never released. Two things
# have to be true before its rows may enter the lake:
#
#   1. Team names must be reconciled onto the canonical football-data.co.uk
#      spelling, or the same match appears twice under two spellings and the
#      league table counts it twice.
#   2. Where both sources have a match, the football-data.co.uk row must win,
#      because it carries shots, cards, xG and odds that this one does not.
# ---------------------------------------------------------------------------

def reconcile_rows(
    rows: list[dict], resolver: TeamResolver, *, strict: bool = True
) -> tuple[list[dict], list[str]]:
    """Rewrite team names onto canonical spellings.

    `strict` drops a row whose club cannot be resolved, which is right when a
    canonical source already covers the competition — an unresolved name there
    means a merge that would corrupt the table. The Champions League has no
    other source for 2026/27, so it runs lenient: an unrecognised club is kept
    under its own name rather than silently vanishing from the competition.
    """
    kept: list[dict] = []
    unresolved: set[str] = set()

    for row in rows:
        home = resolver.resolve(row["home_team"])
        away = resolver.resolve(row["away_team"])

        if strict and (home is None or away is None):
            for original, resolved in ((row["home_team"], home), (row["away_team"], away)):
                if resolved is None:
                    unresolved.add(original)
            continue

        kept.append({
            **row,
            "home_team": home or row["home_team"],
            "away_team": away or row["away_team"],
        })

    return kept, sorted(unresolved)


_ENRICHMENT_COLUMNS = (
    "time", "ht_home_goals", "ht_away_goals",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "home_xg", "away_xg",
    "odds_home", "odds_draw", "odds_away", "odds_over25", "odds_under25",
)


# Lower rank wins a tie. football-data.co.uk is canonical: it carries shots,
# cards, xG and odds, and every team name in the lake is spelled its way.
SOURCE_RANK: dict[str, int] = {
    "football-data.co.uk": 0,
    "openfootball": 1,
    "football-data.org": 2,
}
DEFAULT_SOURCE = "football-data.co.uk"


def dedupe(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse duplicate matches, keeping the best row for each `match_key`.

    "Best" is decided in two steps, and both are needed:

    1. **Richness** — how many enrichment columns are populated. A row with
       shots, cards and odds beats a goals-only row from the gap-filling source.
    2. **Source rank** — the tie-break. Two rows can score identical richness
       (a football-data.org row carrying kick-off time and both half-time scores
       matches an older football-data.co.uk row that predates shot data), and
       ordering must not then depend on the order `build()` happened to append
       them in. Ranking explicitly means a future refactor cannot silently flip
       precedence.

    Previously this was `.unique(subset=["match_key"])`, whose default keep
    strategy is "any" — harmless with one source per match, but with two it
    would non-deterministically discard shots, cards and odds.
    """
    if df.height == 0:
        return df

    richness = pl.sum_horizontal(
        [pl.col(c).is_not_null().cast(pl.Int32) for c in _ENRICHMENT_COLUMNS]
    )
    rank = (
        pl.col("source").replace_strict(SOURCE_RANK, default=len(SOURCE_RANK))
        if "source" in df.columns
        else pl.lit(SOURCE_RANK[DEFAULT_SOURCE])
    )
    return (
        df.with_columns(richness.alias("_richness"), rank.alias("_rank"))
          .sort(["_richness", "_rank"], descending=[True, False], maintain_order=True)
          .unique(subset=["match_key"], keep="first", maintain_order=True)
          .drop("_richness", "_rank")
          .sort("date")
    )


def parse_football_data_org(
    canonical: list[dict],
) -> tuple[list[dict], dict[str, list[str]], int]:
    """Rows from whatever football-data.org payloads are cached on disk.

    Reads the raw layer only — fetching happens in `cli.cmd_refresh`, so
    rebuilding the lake never touches the network.

    Strictness is decided **per season, not per competition**. A season another
    source already covers is reconciled strictly: an unresolvable club there
    means a merge that would double-count a match, so the row is dropped and
    reported. A season no other source has yet — Champions League 2026/27 today
    — runs lenient, because dropping a club nobody else lists would simply erase
    it from the competition.

    Deciding this per season matters: `openfootball.AVAILABLE_CL_SEASONS`
    already asks for 2026-27 on every refresh and currently 404s. The day that
    file appears, this competition-season gains a canonical source and flips to
    strict on its own, with no code change.
    """
    names_by_competition: dict[str, set[str]] = {}
    covered_seasons: set[tuple[str, str]] = set()
    for row in canonical:
        code = row["competition_code"]
        names = names_by_competition.setdefault(code, set())
        names.add(row["home_team"])
        names.add(row["away_team"])
        covered_seasons.add((code, row["season"]))

    out: list[dict] = []
    dropped: dict[str, list[str]] = {}
    rows_dropped = 0

    for code in fdo.COMPETITIONS:
        payload = fdo.read_matches(code)
        if not payload:
            continue

        meta = config.LEAGUES.get(code, {})
        # Spellings are stable across seasons, so the resolver sees every
        # canonical name in the competition regardless of which season it is
        # reconciling.
        resolver = TeamResolver(sorted(names_by_competition.get(code, set())))

        by_season: dict[str, list[dict]] = {}
        for r in fdo.parse_matches(payload, code):
            by_season.setdefault(r["season"], []).append(r)

        for season, season_rows in by_season.items():
            reconciled, unresolved = reconcile_rows(
                season_rows, resolver, strict=(code, season) in covered_seasons)
            rows_dropped += len(season_rows) - len(reconciled)
            if unresolved:
                known = dropped.setdefault(code, [])
                known.extend(n for n in unresolved if n not in known)

            for r in reconciled:
                done = _finish(_row(
                    source="football-data.org",
                    competition_code=code,
                    competition=meta.get("name", code),
                    country=meta.get("country", ""),
                    season=r["season"], date=r["date"], time=r.get("time"),
                    home_team=r["home_team"], away_team=r["away_team"],
                    home_goals=r["home_goals"], away_goals=r["away_goals"],
                    ht_home_goals=r.get("ht_home_goals"),
                    ht_away_goals=r.get("ht_away_goals"),
                ))
                if done:
                    out.append(done)

    return out, dropped, rows_dropped


def build() -> dict[str, int]:
    config.ensure_dirs()
    SILVER_RESULTS.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for season in config.season_codes(20):
        for div in config.TRAINING_DIVISIONS:
            rows.extend(parse_division(season, div))
    for code in config.FD_EXTRA:
        rows.extend(parse_extra(code))
    rows.extend(parse_champions_league())

    # Reconciled against the canonical rows above, so it must come last.
    supplementary, dropped, rows_dropped = parse_football_data_org(rows)
    for code, names in dropped.items():
        log.warning("football-data.org %s: %d unreconciled club(s), %d row(s) "
                    "dropped: %s", code, len(names), rows_dropped, ", ".join(names))
    rows.extend(supplementary)

    if not rows:
        return {"matches": 0, "competitions": 0}

    df = dedupe(pl.DataFrame(rows, schema=ROW_SCHEMA))
    with atomic_path(SILVER_RESULTS / "results.parquet") as temporary:
        df.write_parquet(temporary, compression="zstd")

    return {
        "matches": df.height,
        "competitions": df["competition"].n_unique(),
        "seasons": df["season"].n_unique(),
        "from_football_data_org": len(supplementary),
        "unreconciled_clubs": sum(len(v) for v in dropped.values()),
        "unreconciled_rows_dropped": rows_dropped,
        "with_shots": int(df["home_shots"].is_not_null().sum()),
        "with_xg": int(df["home_xg"].is_not_null().sum()),
        "with_odds": int(df["odds_home"].is_not_null().sum()),
    }
