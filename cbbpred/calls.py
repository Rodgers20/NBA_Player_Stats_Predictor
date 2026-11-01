"""One model call per game: the last prediction logged before tip-off, else the walk-forward backtest call.

Past games the model never logged carry the call it would have made from ratings and a model that
only saw earlier games, labelled ``source: "backtest"``. Games logged before tip-off are ``"live"``.
"""

from __future__ import annotations

import polars as pl
from scipy.stats import norm

from cbbpred import config as cfg
from cbbpred.evaluate import EDGE_CONFIDENCE, confidence
from cbbpred.lines import latest_lines
from cbbpred.predict import market_prob_home
from cbbpred.store import read_json, read_parquet

LINE_COLUMNS = ["game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline", "line_provider",
                "line_fetched_at"]


def _live(games: pl.DataFrame) -> pl.DataFrame:
    preds = read_parquet(cfg.predictions_path())
    if preds.is_empty():
        return preds
    joined = preds.join(games.select("game_id", "kickoff"), on="game_id").filter(pl.col("predicted_at") < pl.col("kickoff"))
    return joined.drop("kickoff").sort("predicted_at").group_by("game_id").last().with_columns(source=pl.lit("live"))


def _backtest(skip: pl.Series) -> pl.DataFrame:
    bt, summary_path = read_parquet(cfg.backtest_path()), cfg.model_dir() / "backtest_summary.json"
    if bt.is_empty() or not summary_path.exists():
        return pl.DataFrame()
    summary = read_json(summary_path)
    sm, st = summary["sigma_margin"], summary["sigma_total"]
    lines = latest_lines()
    b = bt.filter(~pl.col("game_id").is_in(skip.implode())).select(
        "game_id", proj_margin="pred_margin", proj_total="pred_total")
    if lines.height:
        b = b.join(lines.select(LINE_COLUMNS), on="game_id", how="left")
    else:
        b = b.with_columns(**{c: pl.lit(None, dtype=pl.Float64) for c in LINE_COLUMNS[1:5]})
    rows = []
    for g in b.to_dicts():
        m, t, spread, line = g["proj_margin"], g["proj_total"], g["spread_line"], g["total_line"]
        p_home = float(norm.cdf(m / sm))
        p_cover = None if spread is None else float(norm.cdf((m - spread) / sm))
        p_over = None if line is None else float(norm.cdf((t - line) / st))
        rows.append({
            **g, "proj_home_pts": (t + m) / 2, "proj_away_pts": (t - m) / 2, "p_home_win": p_home,
            "p_away_win": 1 - p_home, "win_confidence": confidence(p_home),
            "market_prob_home": market_prob_home(g["home_moneyline"], g["away_moneyline"]),
            "p_home_cover": p_cover, "ats_confidence": None if p_cover is None else confidence(p_cover, EDGE_CONFIDENCE),
            "p_over": p_over, "p_under": None if p_over is None else 1 - p_over,
            "total_confidence": None if p_over is None else confidence(p_over, EDGE_CONFIDENCE),
            "line_source": g["line_provider"], "factors_margin": None, "factors_total": None,
            "model_version": "walk-forward", "source": "backtest", "predicted_at": None,
        })
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def prediction_frame(games: pl.DataFrame) -> pl.DataFrame:
    """Per-game model calls for ``games`` (any status). Empty frame when nothing has been predicted."""
    live = _live(games)
    seen = live["game_id"] if live.height else pl.Series("game_id", [], pl.Utf8)
    parts = [p for p in (live, _backtest(seen)) if p.height]
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
