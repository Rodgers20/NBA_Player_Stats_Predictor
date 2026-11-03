"""Silver layer: conformed, validated, analysis-ready.

Every rule here was derived from an observed anomaly in the bronze layer, not
assumed up front. The three that matter:

1. **Shootout goals are not scoreline goals.** Bronze score reconciliation failed
   on exactly 3 of 51 Euro 2024 matches: England–Switzerland (QF),
   Portugal–France (QF), Portugal–Slovenia (R16). All three carry `period = 5`
   goals — those are penalty shootouts, which decide the tie but do not change
   the 0–0 / 1–1 scoreline. Regulation + extra time is `period <= 4`.

2. **Off-pitch coordinates in 360 data are real, not corrupt.** 24,513 of
   2.7M freeze-frame positions fall outside the 120x80 pitch, ranging
   x -9.5..125.1 and y -9.35..89.5. Footballers genuinely stand off the field:
   throw-ins are taken from behind the touchline, keepers drop behind the goal
   line, players wait offside-deep at corners. These are PRESERVED and flagged,
   never discarded — and the 3D ground plane is sized to accommodate them.

3. **Event coordinates get a rendering clamp, not a data clamp.** One shot sits
   at x = 120.1, a tenth of a yard past the goal line. The true value stays in
   `x`; `render_x`/`render_y` carry a pitch-clamped copy so geometry code never
   has to defend itself.
"""
from __future__ import annotations

import logging

import polars as pl

from .. import config
from ..storage import data_files

log = logging.getLogger(__name__)

SILVER_EVENTS = config.SILVER / "events"
SILVER_FRAMES = config.SILVER / "frames"
SILVER_MATCHES = config.SILVER / "matches"

# Regulation (1,2) + extra time (3,4). Period 5 is the penalty shootout.
REGULATION_PERIODS = (1, 2, 3, 4)
SHOOTOUT_PERIOD = 5

# How far outside the pitch the 3D ground plane must extend, from the observed
# 360 data range with a safety margin.
PITCH_MARGIN = 12.0


def clean_events(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("period") == SHOOTOUT_PERIOD).alias("is_shootout"),
            pl.col("x").clip(0.0, config.PITCH_LENGTH).alias("render_x"),
            pl.col("y").clip(0.0, config.PITCH_WIDTH).alias("render_y"),
            pl.col("pass_end_x").clip(0.0, config.PITCH_LENGTH).alias("render_pass_end_x"),
            pl.col("pass_end_y").clip(0.0, config.PITCH_WIDTH).alias("render_pass_end_y"),
            # A goal that counts toward the scoreline.
            (
                (pl.col("type") == "Shot")
                & (pl.col("shot_outcome") == "Goal")
                & pl.col("period").is_in(REGULATION_PERIODS)
            ).alias("is_scoring_goal"),
            (
                (pl.col("type") == "Pass")
                & pl.col("pass_outcome").is_null()
            ).alias("pass_completed"),
            # Absolute seconds from kick-off, for timeline ordering.
            (pl.col("minute") * 60 + pl.col("second")).alias("clock_seconds"),
            # Distance to the centre of the opposition goal — used for shot geometry.
            (
                (
                    (config.PITCH_LENGTH - pl.col("x")).pow(2)
                    + (pl.col("y") - config.PITCH_WIDTH / 2).pow(2)
                ).sqrt()
            ).alias("distance_to_goal"),
        ]
    )


def clean_frames(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                (pl.col("x") < 0)
                | (pl.col("x") > config.PITCH_LENGTH)
                | (pl.col("y") < 0)
                | (pl.col("y") > config.PITCH_WIDTH)
            ).alias("off_pitch"),
            pl.col("x").clip(-PITCH_MARGIN, config.PITCH_LENGTH + PITCH_MARGIN).alias("render_x"),
            pl.col("y").clip(-PITCH_MARGIN, config.PITCH_WIDTH + PITCH_MARGIN).alias("render_y"),
        ]
    )


def build() -> dict[str, int]:
    config.ensure_dirs()
    for d in (SILVER_EVENTS, SILVER_FRAMES, SILVER_MATCHES):
        d.mkdir(parents=True, exist_ok=True)

    counts = {"events": 0, "frames": 0, "matches": 0}

    for src in data_files(config.BRONZE / "events", "*.parquet"):
        out = clean_events(pl.read_parquet(src))
        out.write_parquet(SILVER_EVENTS / src.name, compression="zstd")
        counts["events"] += out.height

    for src in data_files(config.BRONZE / "frames", "*.parquet"):
        out = clean_frames(pl.read_parquet(src))
        out.write_parquet(SILVER_FRAMES / src.name, compression="zstd")
        counts["frames"] += out.height

    mpath = config.BRONZE / "matches" / "matches.parquet"
    if mpath.exists():
        m = pl.read_parquet(mpath).with_columns(
            (pl.col("home_score") + pl.col("away_score")).alias("total_goals"),
            pl.when(pl.col("home_score") > pl.col("away_score")).then(pl.lit("H"))
            .when(pl.col("home_score") < pl.col("away_score")).then(pl.lit("A"))
            .otherwise(pl.lit("D")).alias("result"),
        )
        m.write_parquet(SILVER_MATCHES / "matches.parquet", compression="zstd")
        counts["matches"] = m.height

    return counts
