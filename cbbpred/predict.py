"""Predict upcoming games and log them (append-only).

The log is never rewritten: a game gets a new row only when its projection or its sportsbook line has
moved, and grading later uses the last row written before tip-off.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import polars as pl
from scipy.stats import norm

from cbbpred import config as cfg
from cbbpred.evaluate import EDGE_CONFIDENCE, confidence
from cbbpred.lines import latest_lines
from cbbpred.model import FEATURES, load_model, load_params
from cbbpred.oddsmath import devig_two_way
from cbbpred.store import read_parquet, write_parquet

log = logging.getLogger(__name__)

MOVE_THRESHOLD = 0.25  # points: smaller projection moves are not worth a new log row


def market_prob_home(home_ml: float | None, away_ml: float | None) -> float | None:
    pair = devig_two_way(home_ml, away_ml)
    return pair[0] if pair else None


def _factor(label: str, value: float) -> dict[str, Any]:
    return {"label": label, "value": round(value, 2)}


def margin_factors(row: dict[str, Any], hfa: float, proj_margin: float) -> list[dict[str, Any]]:
    """Exact decomposition of the projected margin (positive favors the home team)."""
    field = 0.0 if row["neutral"] else hfa
    return [_factor("Home court", field), _factor("Offense edge", row["home_off"] - row["away_off"]),
            _factor("Defense edge", row["away_def"] - row["home_def"]),
            _factor("Rest, form and schedule", proj_margin - row["rating_margin"])]


def total_factors(row: dict[str, Any], base: float, proj_total: float) -> list[dict[str, Any]]:
    return [_factor("Average college game", 2 * base), _factor("Offenses", row["home_off"] + row["away_off"]),
            _factor("Defenses", row["home_def"] + row["away_def"]),
            _factor("Rest, form and schedule", proj_total - row["rating_total"])]


def build_rows(table: pl.DataFrame, lines: pl.DataFrame, now: datetime) -> list[dict[str, Any]]:
    artifact, params = load_model(), load_params()
    upcoming = table.filter((pl.col("status") == "scheduled") & (pl.col("kickoff") > now))
    if upcoming.is_empty():
        return []
    if lines.height:
        upcoming = upcoming.join(lines, on="game_id", how="left")
    x = upcoming.select(FEATURES).to_numpy().astype(float)
    margins, totals = artifact["margin"].predict(x), artifact["total"].predict(x)
    sm, st = artifact["sigma_margin"], artifact["sigma_total"]
    rows = []
    for g, margin, total in zip(upcoming.to_dicts(), margins, totals, strict=True):
        spread, line = g.get("spread_line"), g.get("total_line")
        p_home = float(norm.cdf(margin / sm))
        p_cover = float(norm.cdf((margin - spread) / sm)) if spread is not None else None
        p_over = float(norm.cdf((total - line) / st)) if line is not None else None
        rows.append({
            "game_id": g["game_id"], "predicted_at": now, "model_version": artifact["version"],
            "proj_margin": float(margin), "proj_total": float(total),
            "proj_home_pts": float((total + margin) / 2), "proj_away_pts": float((total - margin) / 2),
            "p_home_win": p_home, "p_away_win": 1 - p_home, "win_confidence": confidence(p_home),
            "market_prob_home": market_prob_home(g.get("home_moneyline"), g.get("away_moneyline")),
            "spread_line": spread, "total_line": line,
            "home_moneyline": g.get("home_moneyline"), "away_moneyline": g.get("away_moneyline"),
            "p_home_cover": p_cover,
            "ats_confidence": None if p_cover is None else confidence(p_cover, EDGE_CONFIDENCE),
            "p_over": p_over, "p_under": None if p_over is None else 1 - p_over,
            "total_confidence": None if p_over is None else confidence(p_over, EDGE_CONFIDENCE),
            "line_source": g.get("line_provider"), "line_fetched_at": g.get("line_fetched_at"),
            "factors_margin": json.dumps(margin_factors(g, params.hfa, float(margin))),
            "factors_total": json.dumps(total_factors(g, params.base, float(total))),
        })
    return rows


def moved(new: dict[str, Any], old: dict[str, Any] | None) -> bool:
    if old is None:
        return True
    if new["spread_line"] != old["spread_line"] or new["total_line"] != old["total_line"]:
        return True
    return (abs(new["proj_margin"] - old["proj_margin"]) > MOVE_THRESHOLD
            or abs(new["proj_total"] - old["proj_total"]) > MOVE_THRESHOLD
            or new["model_version"] != old["model_version"])


def predict_upcoming(now: datetime | None = None) -> pl.DataFrame:
    """Project every unplayed game and append the ones that changed. Returns the rows appended."""
    now = now or datetime.now(UTC)
    table = pl.read_parquet(cfg.ratings_path())
    candidates = build_rows(table, latest_lines(), now)
    path = cfg.predictions_path()
    existing = read_parquet(path)
    last = ({r["game_id"]: r for r in existing.sort("predicted_at").group_by("game_id").last().to_dicts()}
            if existing.height else {})
    fresh = [r for r in candidates if moved(r, last.get(r["game_id"]))]
    if not fresh:
        log.info("no projection moved; nothing appended (%s upcoming)", len(candidates))
        return pl.DataFrame()
    new = pl.DataFrame(fresh, infer_schema_length=None)
    write_parquet(pl.concat([existing, new], how="diagonal_relaxed") if existing.height else new, path)
    log.info("appended %s predictions (%s upcoming)", len(fresh), len(candidates))
    return new
