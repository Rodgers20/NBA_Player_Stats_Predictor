"""Scoring helpers shared by the backtest and the grader: cover sides, records, market comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

EDGE_BUCKETS = (0.0, 1.0, 2.0, 3.0, 4.0)  # points between the model and the line, for the ATS/OU edge table
CAL_BINS = 10
WIN_CONFIDENCE = ((0.80, "high"), (0.65, "medium"), (0.55, "low"))
EDGE_CONFIDENCE = ((0.58, "high"), (0.55, "medium"), (0.53, "low"))


def cover_side(margin: float, line: float | None) -> str | None:
    """Which side covers a home-positive spread: 'home', 'away' or 'push'."""
    if line is None:
        return None
    diff = margin - line
    return "home" if diff > 0 else "away" if diff < 0 else "push"


def total_side(total: float, line: float | None) -> str | None:
    if line is None:
        return None
    return "over" if total > line else "under" if total < line else "push"


def confidence(p_side: float, tiers: tuple[tuple[float, str], ...] = WIN_CONFIDENCE) -> str:
    edge = max(p_side, 1 - p_side)
    return next((label for floor, label in tiers if edge > floor), "coin_flip")


def record(wins: int, n: int) -> dict[str, Any]:
    return {"wins": wins, "losses": n - wins, "n": n, "rate": wins / n if n else None}


def _ats_flags(df: pl.DataFrame) -> np.ndarray:
    """1.0 when the model's side covered, 0.0 when it did not; pushes and model ties are dropped (NaN)."""
    margin, line, pred = (df[c].to_numpy().astype(float) for c in ("margin", "spread_line", "pred_margin"))
    result, pick = np.sign(margin - line), np.sign(pred - line)
    return np.where((result == 0) | (pick == 0), np.nan, (result == pick).astype(float))


def _ou_flags(df: pl.DataFrame) -> np.ndarray:
    total, line, pred = (df[c].to_numpy().astype(float) for c in ("total", "total_line", "pred_total"))
    result, pick = np.sign(total - line), np.sign(pred - line)
    return np.where((result == 0) | (pick == 0), np.nan, (result == pick).astype(float))


def _bucket_table(flags: np.ndarray, edge: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for floor in EDGE_BUCKETS:
        sel = flags[(edge >= floor) & ~np.isnan(flags)]
        rows.append({"min_edge": floor, **record(int(sel.sum()), len(sel))})
    return rows


def vs_market(bt: pl.DataFrame) -> dict[str, Any] | None:
    """Model against the sportsbook on games with both a prediction and a line. None without any lines."""
    spreads = bt.drop_nulls(["spread_line"]) if "spread_line" in bt.columns else bt.clear()
    if spreads.is_empty():
        return None
    ats = _ats_flags(spreads)
    err_model = np.abs(spreads["margin"].to_numpy() - spreads["pred_margin"].to_numpy())
    err_market = np.abs(spreads["margin"].to_numpy() - spreads["spread_line"].to_numpy())
    home_won = spreads["margin"].to_numpy() > 0
    decided = spreads["spread_line"].to_numpy() != 0
    out: dict[str, Any] = {
        "n_games": spreads.height, "seasons": sorted({int(s) for s in spreads["season"].unique()}),
        "model_margin_mae": float(err_model.mean()), "market_margin_mae": float(err_market.mean()),
        "model_winner": record(int(((spreads["pred_margin"].to_numpy() > 0) == home_won)[decided].sum()),
                               int(decided.sum())),
        "market_winner": record(int(((spreads["spread_line"].to_numpy() > 0) == home_won)[decided].sum()),
                                int(decided.sum())),
        "ats": record(int(np.nansum(ats)), int((~np.isnan(ats)).sum())),
        "ats_by_edge": _bucket_table(ats, np.abs(spreads["pred_margin"].to_numpy() - spreads["spread_line"].to_numpy())),
        "by_season": [],
    }
    totals = spreads.drop_nulls(["total_line"])
    if totals.height:
        ou = _ou_flags(totals)
        out["model_total_mae"] = float(np.abs(totals["total"].to_numpy() - totals["pred_total"].to_numpy()).mean())
        out["market_total_mae"] = float(np.abs(totals["total"].to_numpy() - totals["total_line"].to_numpy()).mean())
        out["ou"] = record(int(np.nansum(ou)), int((~np.isnan(ou)).sum()))
        out["ou_by_edge"] = _bucket_table(ou, np.abs(totals["pred_total"].to_numpy() - totals["total_line"].to_numpy()))
    for season in out["seasons"]:
        part = spreads.filter(pl.col("season") == season)
        flags = _ats_flags(part)
        out["by_season"].append({"season": season, "n_games": part.height,
                                 "ats": record(int(np.nansum(flags)), int((~np.isnan(flags)).sum()))})
    return out


def calibration(p: np.ndarray, won: np.ndarray) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, CAL_BINS + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, CAL_BINS - 1)
    return [{"predicted": float(p[idx == b].mean()), "actual": float(won[idx == b].mean()), "n": int((idx == b).sum())}
            for b in range(CAL_BINS) if (idx == b).any()]


def scores(bt: pl.DataFrame, margin_col: str, total_col: str, sigma: float) -> dict[str, float]:
    p = norm.cdf(bt[margin_col].to_numpy() / sigma).clip(1e-6, 1 - 1e-6)
    won = (bt["margin"].to_numpy() > 0).astype(float)
    err = bt["margin"].to_numpy() - bt[margin_col].to_numpy()
    return {
        "margin_mae": float(np.abs(err).mean()), "margin_rmse": float(np.sqrt((err**2).mean())),
        "win_accuracy": float(((p > 0.5) == (won == 1)).mean()),
        "log_loss": float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean()),
        "brier": float(((p - won) ** 2).mean()),
        "total_mae": float(np.abs(bt["total"].to_numpy() - bt[total_col].to_numpy()).mean()),
    }
