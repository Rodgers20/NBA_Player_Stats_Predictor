"""Team statistics for a competition, from the results lake.

The finding this rests on: **the lake already holds per-match team statistics
that nothing was reading.** football-data.co.uk publishes shots, shots on target,
corners, fouls, cards and match xG, and they are populated for the current
domestic seasons:

    Premier League   30 of 37 matches carry shots, on-target, corners and xG
    La Liga          41 of 46
    Champions League  0 of 18   — openfootball publishes results only

So the domestic tabs can show around a dozen real metrics while the Champions
League shows the handful derivable from goals alone. Each competition advertises
what it supports rather than every tab being cut down to the smallest shared set.

Every metric is derived from **one** per-club aggregate, so a twelve-card grid
costs one scan of the lake rather than twelve.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from . import config, league


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    #: False for conceded-style metrics, where the leader is the lowest value.
    higher_is_better: bool = True
    #: Decimal places for display; 0 for counts.
    precision: int = 2
    #: Only offered when the aggregate has this column populated.
    requires: str | None = None


METRICS: tuple[Metric, ...] = (
    Metric("goals_per_match", "Goals per match", "goals", True, 2),
    Metric("conceded_per_match", "Goals conceded per match", "conceded", False, 2),
    Metric("clean_sheets", "Clean sheets", "clean sheets", True, 0),
    Metric("failed_to_score", "Failed to score", "matches", False, 0),
    Metric("xg", "Expected goals (xG)", "xG", True, 2, requires="xg_for"),
    Metric("xg_difference", "xG difference", "xG", True, 2, requires="xg_for"),
    Metric("xg_overperformance", "Goals above xG", "goals", True, 2, requires="xg_for"),
    Metric("shots_per_match", "Shots per match", "shots", True, 1, requires="shots_for"),
    Metric("sot_per_match", "Shots on target per match", "on target", True, 1, requires="sot_for"),
    Metric("shot_accuracy", "Shot accuracy", "%", True, 1, requires="sot_for"),
    Metric("corners_per_match", "Corners per match", "corners", True, 1, requires="corners_for"),
    Metric("fouls_per_match", "Fouls per match", "fouls", False, 1, requires="fouls_for"),
    Metric("cards_per_match", "Cards per match", "cards", False, 2, requires="yellow_for"),
)

_BY_KEY = {m.key: m for m in METRICS}


def _sided(code: str, season_year: int | None) -> pl.DataFrame:
    """One row per club per match, home and away folded together.

    Mirrors the fold `league.standings` uses, so the totals here and the totals
    in the table cannot drift apart.
    """
    df = league._results().filter(pl.col("competition_code") == code)
    used, _fallback = league._resolve_season(code, season_year)
    if used is None or df.height == 0:
        return pl.DataFrame()
    df = df.filter(pl.col("season_year") == used)
    if df.height == 0:
        return pl.DataFrame()

    def side(team: str, gf: str, ga: str, suffix: str, other: str) -> pl.DataFrame:
        return df.select(
            team=pl.col(team),
            gf=pl.col(gf), ga=pl.col(ga),
            xg_for=pl.col(f"{suffix}_xg"), xg_against=pl.col(f"{other}_xg"),
            shots_for=pl.col(f"{suffix}_shots"),
            sot_for=pl.col(f"{suffix}_sot"),
            corners_for=pl.col(f"{suffix}_corners"),
            fouls_for=pl.col(f"{suffix}_fouls"),
            yellow_for=pl.col(f"{suffix}_yellow"),
            red_for=pl.col(f"{suffix}_red"),
        )

    return pl.concat([
        side("home_team", "home_goals", "away_goals", "home", "away"),
        side("away_team", "away_goals", "home_goals", "away", "home"),
    ])


def aggregate(code: str, season_year: int | None = None) -> pl.DataFrame:
    """One row per club, with every total the metrics need."""
    sided = _sided(code, season_year)
    if sided.height == 0:
        return pl.DataFrame()

    return (
        sided.group_by("team")
        .agg(
            played=pl.len(),
            goals_for=pl.col("gf").sum(),
            goals_against=pl.col("ga").sum(),
            clean_sheets=(pl.col("ga") == 0).sum(),
            failed_to_score=(pl.col("gf") == 0).sum(),
            # Averages must divide by matches that *carry* the statistic, not
            # by matches played: 7 of 37 Premier League games have no shot data,
            # and dividing by 37 would understate every club.
            xg_for=pl.col("xg_for").sum(),
            xg_against=pl.col("xg_against").sum(),
            xg_matches=pl.col("xg_for").is_not_null().sum(),
            shots_for=pl.col("shots_for").sum(),
            shots_matches=pl.col("shots_for").is_not_null().sum(),
            sot_for=pl.col("sot_for").sum(),
            corners_for=pl.col("corners_for").sum(),
            corners_matches=pl.col("corners_for").is_not_null().sum(),
            fouls_for=pl.col("fouls_for").sum(),
            fouls_matches=pl.col("fouls_for").is_not_null().sum(),
            yellow_for=pl.col("yellow_for").sum(),
            red_for=pl.col("red_for").sum(),
            cards_matches=pl.col("yellow_for").is_not_null().sum(),
        )
    )


def _value(row: dict, metric: str) -> float | None:
    played = row["played"] or 0
    xg_m = row.get("xg_matches") or 0
    sh_m = row.get("shots_matches") or 0
    co_m = row.get("corners_matches") or 0
    fo_m = row.get("fouls_matches") or 0
    ca_m = row.get("cards_matches") or 0

    if metric == "goals_per_match":
        return row["goals_for"] / played if played else None
    if metric == "conceded_per_match":
        return row["goals_against"] / played if played else None
    if metric == "clean_sheets":
        return row["clean_sheets"]
    if metric == "failed_to_score":
        return row["failed_to_score"]
    if metric == "xg":
        return row["xg_for"] if xg_m else None
    if metric == "xg_difference":
        return (row["xg_for"] - row["xg_against"]) if xg_m else None
    if metric == "xg_overperformance":
        return (row["goals_for"] - row["xg_for"]) if xg_m else None
    if metric == "shots_per_match":
        return row["shots_for"] / sh_m if sh_m else None
    if metric == "sot_per_match":
        return row["sot_for"] / sh_m if sh_m else None
    if metric == "shot_accuracy":
        return 100 * row["sot_for"] / row["shots_for"] if row.get("shots_for") else None
    if metric == "corners_per_match":
        return row["corners_for"] / co_m if co_m else None
    if metric == "fouls_per_match":
        return row["fouls_for"] / fo_m if fo_m else None
    if metric == "cards_per_match":
        return ((row["yellow_for"] + row["red_for"]) / ca_m) if ca_m else None
    return None


def available_metrics(code: str, season_year: int | None = None) -> list[dict]:
    """Which metrics this competition can actually fill.

    A card is offered only when its underlying column is populated, so the
    Champions League tab shows five real cards rather than thirteen with eight
    empty.
    """
    agg = aggregate(code, season_year)
    if agg.height == 0:
        return []
    out = []
    for m in METRICS:
        if m.requires and int(agg[m.requires].is_not_null().sum()) == 0:
            continue
        if m.requires and float(agg[m.requires].fill_null(0).sum()) == 0:
            continue
        out.append({"key": m.key, "label": m.label, "unit": m.unit,
                    "higher_is_better": m.higher_is_better, "precision": m.precision})
    return out


def leaders(code: str, metric: str, season_year: int | None = None,
            limit: int = 10) -> dict:
    """One ranked board. Empty when the competition has no data for it."""
    spec = _BY_KEY.get(metric)
    meta = config.LEAGUES.get(code, {})
    result = {
        "competition_code": code,
        "competition": meta.get("name", code),
        "metric": metric,
        "label": spec.label if spec else metric,
        "unit": spec.unit if spec else "",
        "higher_is_better": spec.higher_is_better if spec else True,
        "precision": spec.precision if spec else 2,
        "rows": [],
        "note": None,
    }
    if spec is None:
        result["note"] = f"Unknown metric {metric!r}."
        return result

    agg = aggregate(code, season_year)
    if agg.height == 0:
        result["note"] = "No matches recorded for this competition yet."
        return result

    scored = []
    for row in agg.to_dicts():
        value = _value(row, metric)
        if value is None:
            continue
        scored.append({"team": row["team"], "value": round(float(value), spec.precision),
                       "played": row["played"]})

    if not scored:
        result["note"] = (
            f"{result['competition']} does not publish the data behind "
            f"{spec.label.lower()}."
        )
        return result

    scored.sort(key=lambda r: r["value"], reverse=spec.higher_is_better)
    result["rows"] = scored[:limit]

    # Say how many matches back the average, since coverage is partial.
    if spec.requires:
        covered = int(agg[spec.requires].is_not_null().sum())
        total = agg.height
        if covered < total:
            result["note"] = f"Based on the {covered} of {total} clubs with this data."
    return result


def club(team: str, code: str, season_year: int | None = None) -> dict:
    """One club's season, metric by metric, with where it ranks.

    The club page needs the same numbers the competition tab shows, but read
    down a column instead of across a row — and a bare "1.4 shots on target" is
    only meaningful next to what the rest of the division manages, so every
    metric carries its rank.

    The whole thing costs one scan: `aggregate` is called once and every metric
    is derived from the same frame.
    """
    meta = config.LEAGUES.get(code, {})
    out: dict = {
        "competition_code": code,
        "competition": meta.get("name", code),
        "team": team,
        "played": 0,
        "metrics": [],
        "note": None,
    }

    agg = aggregate(code, season_year)
    if agg.height == 0:
        out["note"] = "No matches recorded for this competition yet."
        return out

    rows = agg.to_dicts()
    mine = next((r for r in rows if r["team"] == team), None)
    if mine is None:
        out["note"] = f"{team} has no matches in this competition yet."
        return out
    out["played"] = int(mine["played"])

    for spec in METRICS:
        value = _value(mine, spec.key)
        if value is None:
            continue
        # Rank against every club that has the metric at all, so a division
        # with partial coverage ranks within what is actually published.
        peers = [v for v in (_value(r, spec.key) for r in rows) if v is not None]
        ahead = (lambda v: v > value) if spec.higher_is_better else (lambda v: v < value)
        better = sum(1 for v in peers if ahead(v))
        out["metrics"].append({
            "key": spec.key,
            "label": spec.label,
            "unit": spec.unit,
            "value": round(float(value), spec.precision),
            "precision": spec.precision,
            "higher_is_better": spec.higher_is_better,
            "rank": better + 1,
            "of": len(peers),
        })

    if not out["metrics"]:
        out["note"] = f"{out['competition']} publishes no per-match statistics."
    return out
