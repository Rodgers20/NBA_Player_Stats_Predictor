"""Feature engineering for match prediction.

The single most important property here is **no leakage**. A feature for a match
played on date T may only use information available strictly before T. Every
rolling statistic is computed on a team's match history and then shifted by one,
so a team's form entering a game never includes that game. Elo records the rating
as it stood before kick-off. Head-to-head excludes the current meeting.

The second is that the **market is a feature, not a cheat**. Closing odds are
published before kick-off, so implied probabilities are legitimately available at
prediction time — and they are a strong baseline any model should be measured
against rather than quietly compared away from.

Features are organised into GROUPS so the tuner can ablate them honestly on a
validation window instead of us assuming which ones help.
"""
from __future__ import annotations

import polars as pl

from .. import config
from . import elo as elo_mod
from . import style

ROLL_WINDOWS = (5, 10)
ROLL_STATS = ("ppg", "gf", "ga", "shots", "sot", "shots_against", "corners")


def _to_team_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Explode each match into two team-perspective rows."""
    common = [
        pl.col("match_key"), pl.col("competition_code"), pl.col("season"), pl.col("date"),
    ]
    home = df.select(
        *common,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("home_goals").alias("gf"), pl.col("away_goals").alias("ga"),
        pl.col("home_shots").alias("shots"), pl.col("away_shots").alias("shots_against"),
        pl.col("home_sot").alias("sot"), pl.col("away_sot").alias("sot_against"),
        pl.col("home_corners").alias("corners"),
    )
    away = df.select(
        *common,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        pl.col("away_goals").alias("gf"), pl.col("home_goals").alias("ga"),
        pl.col("away_shots").alias("shots"), pl.col("home_shots").alias("shots_against"),
        pl.col("away_sot").alias("sot"), pl.col("home_sot").alias("sot_against"),
        pl.col("away_corners").alias("corners"),
    )
    return pl.concat([home, away]).with_columns(
        pl.when(pl.col("gf") > pl.col("ga")).then(3)
        .when(pl.col("gf") == pl.col("ga")).then(1)
        .otherwise(0).alias("points"),
    )


def _rolling(team_rows: pl.DataFrame) -> pl.DataFrame:
    """Rolling form per team, shifted so the current match is excluded."""
    from collections import defaultdict
    from datetime import date
    from itertools import groupby

    # Snapshot all rows on a date before updating history. Unplayed fixtures
    # never occupy a rolling-window slot or count as a loss/rest-day reset.
    history = defaultdict(list)
    output = []
    for day, batch in groupby(team_rows.sort(["date", "match_key"]).to_dicts(),
                              key=lambda r: r["date"]):
        batch = list(batch)
        for row in batch:
            past = history[row["team"]]
            values = dict(row)
            def mean(rows, col):
                vals = [r[col] for r in rows if r[col] is not None]
                return sum(vals) / len(vals) if len(vals) >= 2 else None
            for w in ROLL_WINDOWS:
                for col in ("points", "gf", "ga", "shots", "sot", "shots_against", "corners"):
                    alias = "ppg" if col == "points" else col
                    values[f"{alias}_r{w}"] = mean(past[-w:], col)
            values["rest_days"] = ((date.fromisoformat(day) - date.fromisoformat(past[-1]["date"])).days
                                   if past else None)
            values["team_match_no"] = len(past) + 1
            values["last_match_date"] = past[-1]["date"] if past else None
            season = [r for r in past if r["season"] == row["season"]]
            values["season_matches"] = len(season)
            values["season_ppg"] = sum(r["points"] for r in season) / len(season) if season else None
            venue = [r for r in past if r["is_home"] == row["is_home"]][-6:]
            for col, alias in (("points", "ppg"), ("gf", "gf"), ("ga", "ga")):
                values[f"venue_{alias}"] = mean(venue, col)
            output.append(values)
        for row in batch:
            if row["gf"] is not None and row["ga"] is not None:
                history[row["team"]].append(row)
    return pl.DataFrame(output, infer_schema_length=None)


def _implied(df: pl.DataFrame) -> pl.DataFrame:
    """Market-implied probabilities, de-vigged by normalising the overround."""
    inv_h = 1.0 / pl.col("odds_home")
    inv_d = 1.0 / pl.col("odds_draw")
    inv_a = 1.0 / pl.col("odds_away")
    total = inv_h + inv_d + inv_a
    return df.with_columns(
        (inv_h / total).alias("mkt_home"),
        (inv_d / total).alias("mkt_draw"),
        (inv_a / total).alias("mkt_away"),
        total.alias("mkt_overround"),
    )


def _head_to_head(df: pl.DataFrame) -> pl.DataFrame:
    """Average goal difference in previous meetings of the same two clubs.

    Ordered pair-independent: Arsenal-Chelsea and Chelsea-Arsenal share a key,
    with the goal difference always expressed from the current home side's view.
    """
    rows = df.sort(["date", "match_key"]).to_dicts()
    history: dict[tuple[str, str], list[tuple[str, int, str]]] = {}
    h2h_gd: list[float | None] = []
    h2h_n: list[int] = []

    for r in rows:
        h, a = r["home_team"], r["away_team"]
        key = tuple(sorted((h, a)))  # type: ignore[assignment]
        past = [item for item in history.get(key, []) if item[2] < r["date"]]
        if past:
            # goal difference from the perspective of the current home side
            diffs = [gd if winner_home == h else -gd for winner_home, gd, _ in past]
            h2h_gd.append(sum(diffs[-5:]) / len(diffs[-5:]))
            h2h_n.append(len(past))
        else:
            h2h_gd.append(None)
            h2h_n.append(0)

        hg, ag = r["home_goals"], r["away_goals"]
        if hg is not None and ag is not None:
            history.setdefault(key, []).append((h, hg - ag, r["date"]))

    return df.sort(["date", "match_key"]).with_columns(
        pl.Series("h2h_gd", h2h_gd, dtype=pl.Float64),
        pl.Series("h2h_meetings", h2h_n, dtype=pl.Int32),
    )


# --------------------------------------------------------------------------
# Feature groups — the tuner ablates over these
# --------------------------------------------------------------------------

GROUPS: dict[str, list[str]] = {
    "style": style.COLUMNS,
    "form": (
        [f"home_{s}_r{w}" for w in ROLL_WINDOWS for s in ROLL_STATS]
        + [f"away_{s}_r{w}" for w in ROLL_WINDOWS for s in ROLL_STATS]
        + ["home_season_ppg", "away_season_ppg", "home_rest_days", "away_rest_days"]
    ),
    "elo": ["home_elo", "away_elo", "elo_diff", "elo_expected_home"],
    "venue": ["home_venue_ppg", "home_venue_gf", "home_venue_ga",
              "away_venue_ppg", "away_venue_gf", "away_venue_ga"],
    "h2h": ["h2h_gd", "h2h_meetings"],
    "competition": ["competition_idx"],
    "market": ["mkt_home", "mkt_draw", "mkt_away"],
}

CATEGORICAL = {"competition_idx"}


def all_feature_columns(groups: list[str] | None = None) -> list[str]:
    keys = groups if groups is not None else list(GROUPS)
    out: list[str] = []
    for k in keys:
        out.extend(GROUPS[k])
    return out


# Back-compat for code that imported the flat list.
FEATURE_COLUMNS = all_feature_columns()


def build(
    min_history: int = 6,
    extra_fixtures: list[dict] | None = None,
    include_style: bool = False,
) -> pl.DataFrame:
    """Assemble the modelling table with every candidate feature attached.

    `extra_fixtures` appends unplayed matches (null goals) before feature
    computation. This is how prediction works at serve time: a fixture is pushed
    through the SAME code path that produced the training features, so there is
    no possibility of train/serve skew. Reconstructing features by hand on the
    serving side is the classic source of silent model bugs.

    Unplayed rows carry `is_fixture = True`, are exempt from the cold-start
    filter, and never contribute to Elo, head-to-head or rolling history —
    they have no result to learn from.
    """
    src = config.SILVER / "results" / "results.parquet"
    df = pl.read_parquet(src).filter(
        pl.col("date").is_not_null() & pl.col("home_team").is_not_null()
    ).with_columns(pl.lit(False).alias("is_fixture"))

    if extra_fixtures:
        rows = []
        for f in extra_fixtures:
            row = {c: None for c in df.columns}
            row.update(f)
            row["is_fixture"] = True
            row["match_key"] = (
                f"FIXTURE|{f.get('competition_code')}|{f.get('date')}|"
                f"{f.get('home_team')}|{f.get('away_team')}"
            )
            rows.append(row)
        add = pl.DataFrame(rows, schema=df.schema)
        df = pl.concat([df, add], how="vertical")

    if include_style:
        df = style.attach(df)
    df = elo_mod.attach(df)
    df = _head_to_head(df)

    tr = _rolling(_to_team_rows(df))

    roll_cols = [f"{s}_r{w}" for w in ROLL_WINDOWS for s in ROLL_STATS]
    extra = ["rest_days", "team_match_no", "season_ppg", "last_match_date", "season_matches",
             "venue_ppg", "venue_gf", "venue_ga"]
    keep = ["match_key", *roll_cols, *extra]

    home_side = tr.filter(pl.col("is_home")).select(keep).rename(
        {c: f"home_{c}" for c in keep if c != "match_key"})
    away_side = tr.filter(~pl.col("is_home")).select(keep).rename(
        {c: f"away_{c}" for c in keep if c != "match_key"})

    out = (
        df.join(home_side, on="match_key", how="left")
          .join(away_side, on="match_key", how="left")
    )
    out = _implied(out)

    # Competition as an explicit categorical: scoring rates and home advantage
    # differ materially between, say, MLS and Serie A, and the model should not
    # have to rediscover that from form alone.
    codes = sorted(out["competition_code"].unique().to_list())
    mapping = {c: i for i, c in enumerate(codes)}
    out = out.with_columns(
        pl.col("competition_code").replace_strict(mapping, default=-1)
        .cast(pl.Int32).alias("competition_idx")
    )

    out = out.filter(
        pl.col("is_fixture")
        | (
            (pl.col("home_team_match_no") >= min_history)
            & (pl.col("away_team_match_no") >= min_history)
        )
    )
    return out.sort(["date", "match_key"])


def feature_matrix(
    df: pl.DataFrame, groups: list[str] | None = None
) -> tuple[pl.DataFrame, list[str]]:
    """Select usable feature columns for the requested groups."""
    cols = [c for c in all_feature_columns(groups) if c in df.columns]
    usable = [c for c in cols if df[c].is_not_null().sum() > 0]
    return df.select(usable), usable


def categorical_indices(cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(cols) if c in CATEGORICAL]
