"""Grade model calls against final scores.

Each game is graded on its pre-tip call (see ``calls``), so a late projection can never be rewritten with
hindsight. Against-the-spread and over/under results exist only for games whose line was captured.
Pushes are recorded as ``push`` and left out of every win/loss record.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from cbbpred import config as cfg
from cbbpred.calls import prediction_frame
from cbbpred.evaluate import cover_side, record, total_side
from cbbpred.store import write_parquet


def _market_pick_home(g: dict[str, Any]) -> bool | None:
    """Whether the market favors the home side: moneyline first, else the spread; None for a pick'em."""
    if g.get("market_prob_home") is not None and g["market_prob_home"] != 0.5:
        return g["market_prob_home"] > 0.5
    spread = g.get("spread_line")
    return None if spread in (None, 0) else spread > 0


def grade_row(g: dict[str, Any]) -> dict[str, Any]:
    margin = g["home_score"] - g["away_score"]
    total = g["home_score"] + g["away_score"]
    home_won = margin > 0
    cover, ou = cover_side(margin, g["spread_line"]), total_side(total, g["total_line"])
    pick_home_cover = None if g["p_home_cover"] is None else g["p_home_cover"] >= 0.5
    pick_over = None if g["p_over"] is None else g["p_over"] >= 0.5
    market_home = _market_pick_home(g)
    return {
        **g, "margin": margin, "total_points": total, "winner": "home" if home_won else "away",
        "pick_winner": "home" if g["p_home_win"] >= 0.5 else "away",
        "winner_correct": (g["p_home_win"] >= 0.5) == home_won,
        "market_winner_correct": None if market_home is None else market_home == home_won,
        "ats_result": cover, "pick_ats": None if pick_home_cover is None else "home" if pick_home_cover else "away",
        "ats_correct": None if cover in (None, "push") or pick_home_cover is None else (cover == "home") == pick_home_cover,
        "total_result": ou, "pick_total": None if pick_over is None else "over" if pick_over else "under",
        "total_correct": None if ou in (None, "push") or pick_over is None else (ou == "over") == pick_over,
        "margin_error": abs(margin - g["proj_margin"]), "total_error": abs(total - g["proj_total"]),
    }


def grade(write: bool = True) -> pl.DataFrame:
    games = pl.read_parquet(cfg.games_path())
    finals = games.filter((pl.col("status") == "final") & pl.col("home_score").is_not_null()).select(
        "game_id", "season", "kickoff", "gameday", "home_id", "away_id", "home_abbr", "away_abbr", "home_score", "away_score")
    calls = prediction_frame(games)
    if calls.is_empty():
        return pl.DataFrame()
    rows = [grade_row(g) for g in finals.join(calls, on="game_id").to_dicts()]
    out = pl.DataFrame(rows, infer_schema_length=None).sort("kickoff") if rows else pl.DataFrame()
    if write and out.height:
        write_parquet(out, cfg.graded_path())
    return out


def _record(df: pl.DataFrame, col: str) -> dict[str, Any]:
    graded = df.drop_nulls(col)
    return record(int(graded[col].sum()) if graded.height else 0, graded.height)


def track_record(graded: pl.DataFrame) -> dict[str, Any]:
    if graded.is_empty():
        return {"games": 0, "winner": None, "market_winner": None, "ats": None, "total": None,
                "margin_mae": None, "total_mae": None}
    return {"games": graded.height, "winner": _record(graded, "winner_correct"),
            "market_winner": _record(graded, "market_winner_correct"),
            "ats": _record(graded, "ats_correct"), "total": _record(graded, "total_correct"),
            "margin_mae": float(graded["margin_error"].mean()), "total_mae": float(graded["total_error"].mean())}


def track_record_by_season(graded: pl.DataFrame) -> list[dict[str, Any]]:
    if graded.is_empty():
        return []
    return [{"season": int(s), **track_record(r)} for (s,), r in sorted(graded.group_by("season"), key=lambda g: g[0][0])]
