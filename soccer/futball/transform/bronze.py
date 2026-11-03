"""Bronze layer: raw JSON -> typed, columnar Parquet. One row per event.

Bronze is a faithful, *typed* projection of the raw payload. It flattens the
StatsBomb nested structure into flat columns and nothing more — no business
logic, no derived metrics, no filtering. That separation is what makes the
lineage auditable: if a gold number looks wrong we can walk it back to bronze
and then to the exact raw bytes.

Parquet matters here for a concrete reason: the T7 link measured ~37 MB/s, so
I/O is the binding constraint. A 3.1 MB events JSON compresses to a few hundred
KB of columnar Parquet that DuckDB can scan by column, which turns a
disk-bound workload into a cache-friendly one.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from .. import config
from ..sources import statsbomb

log = logging.getLogger(__name__)

BRONZE_EVENTS = config.BRONZE / "events"
BRONZE_FRAMES = config.BRONZE / "frames"
BRONZE_MATCHES = config.BRONZE / "matches"
BRONZE_LINEUPS = config.BRONZE / "lineups"


def _loc(value: Any, index: int) -> float | None:
    """Pull component `index` out of a StatsBomb [x, y] or [x, y, z] location."""
    if isinstance(value, (list, tuple)) and len(value) > index:
        v = value[index]
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _nested(d: dict, *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def flatten_events(match_id: int, events: list[dict]) -> pl.DataFrame:
    """Flatten one match's events into a typed frame."""
    rows: list[dict[str, Any]] = []
    for e in events:
        shot = e.get("shot") or {}
        p = e.get("pass") or {}
        rows.append(
            {
                "match_id": match_id,
                "event_id": e.get("id"),
                "index": e.get("index"),
                "period": e.get("period"),
                "minute": e.get("minute"),
                "second": e.get("second"),
                "timestamp": e.get("timestamp"),
                "duration": e.get("duration"),
                "type": _nested(e, "type", "name"),
                "play_pattern": _nested(e, "play_pattern", "name"),
                "possession": e.get("possession"),
                "possession_team": _nested(e, "possession_team", "name"),
                "team_id": _nested(e, "team", "id"),
                "team": _nested(e, "team", "name"),
                "player_id": _nested(e, "player", "id"),
                "player": _nested(e, "player", "name"),
                "position": _nested(e, "position", "name"),
                "x": _loc(e.get("location"), 0),
                "y": _loc(e.get("location"), 1),
                "under_pressure": bool(e.get("under_pressure", False)),
                "out": bool(e.get("out", False)),
                # --- pass ---
                "pass_end_x": _loc(p.get("end_location"), 0),
                "pass_end_y": _loc(p.get("end_location"), 1),
                "pass_length": p.get("length"),
                "pass_angle": p.get("angle"),
                "pass_height": _nested(p, "height", "name"),
                "pass_body_part": _nested(p, "body_part", "name"),
                "pass_type": _nested(p, "type", "name"),
                "pass_recipient_id": _nested(p, "recipient", "id"),
                "pass_recipient": _nested(p, "recipient", "name"),
                "pass_outcome": _nested(p, "outcome", "name"),
                "pass_cross": bool(p.get("cross", False)),
                "pass_switch": bool(p.get("switch", False)),
                "pass_assisted_shot_id": p.get("assisted_shot_id"),
                "pass_shot_assist": bool(p.get("shot_assist", False)),
                "pass_goal_assist": bool(p.get("goal_assist", False)),
                # --- shot ---
                "shot_xg": shot.get("statsbomb_xg"),
                "shot_outcome": _nested(shot, "outcome", "name"),
                "shot_technique": _nested(shot, "technique", "name"),
                "shot_body_part": _nested(shot, "body_part", "name"),
                "shot_type": _nested(shot, "type", "name"),
                "shot_first_time": bool(shot.get("first_time", False)),
                "shot_end_x": _loc(shot.get("end_location"), 0),
                "shot_end_y": _loc(shot.get("end_location"), 1),
                "shot_end_z": _loc(shot.get("end_location"), 2),
                "shot_freeze_frame_n": len(shot.get("freeze_frame") or []),
                # --- carry / dribble / duel outcomes ---
                "carry_end_x": _loc(_nested(e, "carry", "end_location"), 0),
                "carry_end_y": _loc(_nested(e, "carry", "end_location"), 1),
                "dribble_outcome": _nested(e, "dribble", "outcome", "name"),
                "duel_outcome": _nested(e, "duel", "outcome", "name"),
                "interception_outcome": _nested(e, "interception", "outcome", "name"),
                "goalkeeper_type": _nested(e, "goalkeeper", "type", "name"),
                "goalkeeper_outcome": _nested(e, "goalkeeper", "outcome", "name"),
                "foul_committed_card": _nested(e, "foul_committed", "card", "name"),
                "bad_behaviour_card": _nested(e, "bad_behaviour", "card", "name"),
                "substitution_replacement": _nested(
                    e, "substitution", "replacement", "name"
                ),
            }
        )

    schema = {
        "match_id": pl.Int64, "event_id": pl.Utf8, "index": pl.Int64,
        "period": pl.Int32, "minute": pl.Int32, "second": pl.Int32,
        "timestamp": pl.Utf8, "duration": pl.Float64, "type": pl.Utf8,
        "play_pattern": pl.Utf8, "possession": pl.Int64, "possession_team": pl.Utf8,
        "team_id": pl.Int64, "team": pl.Utf8, "player_id": pl.Int64,
        "player": pl.Utf8, "position": pl.Utf8, "x": pl.Float64, "y": pl.Float64,
        "under_pressure": pl.Boolean, "out": pl.Boolean,
        "pass_end_x": pl.Float64, "pass_end_y": pl.Float64, "pass_length": pl.Float64,
        "pass_angle": pl.Float64, "pass_height": pl.Utf8, "pass_body_part": pl.Utf8,
        "pass_type": pl.Utf8, "pass_recipient_id": pl.Int64, "pass_recipient": pl.Utf8,
        "pass_outcome": pl.Utf8, "pass_cross": pl.Boolean, "pass_switch": pl.Boolean,
        "pass_assisted_shot_id": pl.Utf8, "pass_shot_assist": pl.Boolean,
        "pass_goal_assist": pl.Boolean,
        "shot_xg": pl.Float64, "shot_outcome": pl.Utf8, "shot_technique": pl.Utf8,
        "shot_body_part": pl.Utf8, "shot_type": pl.Utf8, "shot_first_time": pl.Boolean,
        "shot_end_x": pl.Float64, "shot_end_y": pl.Float64, "shot_end_z": pl.Float64,
        "shot_freeze_frame_n": pl.Int32,
        "carry_end_x": pl.Float64, "carry_end_y": pl.Float64,
        "dribble_outcome": pl.Utf8, "duel_outcome": pl.Utf8,
        "interception_outcome": pl.Utf8, "goalkeeper_type": pl.Utf8,
        "goalkeeper_outcome": pl.Utf8, "foul_committed_card": pl.Utf8,
        "bad_behaviour_card": pl.Utf8, "substitution_replacement": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def flatten_frames(match_id: int, frames: list[dict]) -> pl.DataFrame:
    """Flatten 360 freeze-frames: one row per player per frame."""
    rows: list[dict[str, Any]] = []
    for f in frames:
        event_uuid = f.get("event_uuid")
        visible = f.get("visible_area") or []
        for i, pl_ in enumerate(f.get("freeze_frame") or []):
            rows.append(
                {
                    "match_id": match_id,
                    "event_id": event_uuid,
                    "player_index": i,
                    "x": _loc(pl_.get("location"), 0),
                    "y": _loc(pl_.get("location"), 1),
                    "teammate": bool(pl_.get("teammate", False)),
                    "actor": bool(pl_.get("actor", False)),
                    "keeper": bool(pl_.get("keeper", False)),
                    "visible_area_n": len(visible) // 2,
                }
            )
    schema = {
        "match_id": pl.Int64, "event_id": pl.Utf8, "player_index": pl.Int32,
        "x": pl.Float64, "y": pl.Float64, "teammate": pl.Boolean,
        "actor": pl.Boolean, "keeper": pl.Boolean, "visible_area_n": pl.Int32,
    }
    return pl.DataFrame(rows, schema=schema)


def flatten_matches(matches: list[dict]) -> pl.DataFrame:
    rows = []
    for m in matches:
        rows.append(
            {
                "match_id": m.get("match_id"),
                "competition_id": _nested(m, "competition", "competition_id"),
                "competition": _nested(m, "competition", "competition_name"),
                "country": _nested(m, "competition", "country_name"),
                "season_id": _nested(m, "season", "season_id"),
                "season": _nested(m, "season", "season_name"),
                "match_date": m.get("match_date"),
                "kick_off": m.get("kick_off"),
                "stadium": _nested(m, "stadium", "name"),
                "referee": _nested(m, "referee", "name"),
                "home_team_id": _nested(m, "home_team", "home_team_id"),
                "home_team": _nested(m, "home_team", "home_team_name"),
                "away_team_id": _nested(m, "away_team", "away_team_id"),
                "away_team": _nested(m, "away_team", "away_team_name"),
                "home_score": m.get("home_score"),
                "away_score": m.get("away_score"),
                "match_week": m.get("match_week"),
                "competition_stage": _nested(m, "competition_stage", "name"),
                "home_manager": (_nested(m, "home_team", "managers") or [{}])[0].get("name")
                if _nested(m, "home_team", "managers") else None,
                "away_manager": (_nested(m, "away_team", "managers") or [{}])[0].get("name")
                if _nested(m, "away_team", "managers") else None,
            }
        )
    schema = {
        "match_id": pl.Int64, "competition_id": pl.Int64, "competition": pl.Utf8,
        "country": pl.Utf8, "season_id": pl.Int64, "season": pl.Utf8,
        "match_date": pl.Utf8, "kick_off": pl.Utf8, "stadium": pl.Utf8,
        "referee": pl.Utf8, "home_team_id": pl.Int64, "home_team": pl.Utf8,
        "away_team_id": pl.Int64, "away_team": pl.Utf8, "home_score": pl.Int32,
        "away_score": pl.Int32, "match_week": pl.Int32, "competition_stage": pl.Utf8,
        "home_manager": pl.Utf8, "away_manager": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def build(match_ids: list[int] | None = None) -> dict[str, int]:
    """Build the bronze layer from whatever raw data is on disk."""
    config.ensure_dirs()
    BRONZE_EVENTS.mkdir(parents=True, exist_ok=True)
    BRONZE_FRAMES.mkdir(parents=True, exist_ok=True)
    BRONZE_MATCHES.mkdir(parents=True, exist_ok=True)

    ids = match_ids if match_ids is not None else statsbomb.local_match_ids()
    counts = {"matches": 0, "events": 0, "frames": 0, "event_files": 0, "frame_files": 0}

    matches = statsbomb.all_local_matches()
    if matches:
        mdf = flatten_matches(matches).unique(subset=["match_id"])
        mdf.write_parquet(BRONZE_MATCHES / "matches.parquet", compression="zstd")
        counts["matches"] = mdf.height

    for mid in ids:
        try:
            edf = flatten_events(mid, statsbomb.read_events(mid))
            edf.write_parquet(
                BRONZE_EVENTS / f"match_id={mid}.parquet", compression="zstd"
            )
            counts["events"] += edf.height
            counts["event_files"] += 1

            frames = statsbomb.read_360(mid)
            if frames:
                fdf = flatten_frames(mid, frames)
                fdf.write_parquet(
                    BRONZE_FRAMES / f"match_id={mid}.parquet", compression="zstd"
                )
                counts["frames"] += fdf.height
                counts["frame_files"] += 1
        except Exception as exc:  # noqa: BLE001
            log.error("bronze build failed for match %s: %s", mid, exc)
            raise

    return counts
