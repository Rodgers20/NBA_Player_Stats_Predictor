"""Standings, model performance and health payloads."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nhlpred import config as cfg
from nhlpred import game_model as gm
from nhlpred import predict as pr
from nhlpred import prop_model as pm
from nhlpred.context import Context, team_ref
from nhlpred.views_props import LINEUP_NOTE

RECENT_PICKS = 25


def standings(ctx: Context) -> dict[str, Any]:
    by_abbr = {t["abbr"]: t["id"] for t in ctx.teams.values()}
    tree: dict[str, dict[str, list]] = {}
    for r in ctx.standings:
        abbr = r["teamAbbrev"]["default"]
        tid = by_abbr.get(abbr)
        if tid is None:
            continue
        tree.setdefault(r["conferenceName"], {}).setdefault(r["divisionName"], []).append({
            "team": team_ref(ctx, tid), "rank": int(r["divisionSequence"]), "gp": r["gamesPlayed"], "w": r["wins"], "l": r["losses"],
            "otl": r["otLosses"], "pts": r["points"], "pct": round(r["pointPctg"], 3), "gf": r["goalFor"], "ga": r["goalAgainst"],
            "diff": r["goalDifferential"], "l10": f"{r['l10Wins']}-{r['l10Losses']}-{r['l10OtLosses']}",
            "streak": f"{r.get('streakCode', '')}{r.get('streakCount', '')}", "wildcard": r.get("wildcardSequence"),
            "elo": round(ctx.elo_now.get(tid, 1500.0))})
    return {"generated_at": ctx.stamp, "conferences": [
        {"name": c, "divisions": [{"name": d, "teams": sorted(ts, key=lambda t: t["rank"])} for d, ts in sorted(divs.items())]}
        for c, divs in sorted(tree.items())]}


def _recent(ctx: Context) -> list[dict[str, Any]]:
    g = ctx.data.games
    fin = g[g.status == "final"].set_index("game_id")
    p = ctx.gpred[ctx.gpred.index.isin(fin.index)]
    rows = []
    for gid, r in p.join(fin[["date", "home_abbr", "away_abbr", "home_score", "away_score"]]).sort_values("date").tail(RECENT_PICKS).iterrows():
        pick_home = r.p_home >= 0.5
        won_home = r.home_score > r.away_score
        rows.append({"date": r.date, "game_id": int(gid), "pick": r.home_abbr if pick_home else r.away_abbr,
                     "p": round(float(max(r.p_home, 1 - r.p_home)), 4), "result": r.home_abbr if won_home else r.away_abbr,
                     "correct": bool(pick_home == won_home), "source": r.source})
    return rows[::-1]


def performance(ctx: Context) -> dict[str, Any]:
    games_summary, props_summary = gm.summary_json(), pm.summary_json()
    graded = pr.grade_games(ctx.data.games)
    live = {"graded": int(len(graded)), "correct": int(graded.correct.sum()) if len(graded) else 0}
    live["rate"] = round(live["correct"] / live["graded"], 4) if live["graded"] else None
    return {"generated_at": ctx.stamp,
            "game": games_summary.get("backtest", {}),
            "props": props_summary.get("backtest", {}),
            "record": {"games": {"live": live, "note": "live = calls frozen before puck drop; recent[] also shows walk-forward replays (source=backtest)."},
                       "props": pr.props_record(pr.grade_props(ctx.data.skaters)), "recent": _recent(ctx)}}


def health(ctx: Context, days: list[dict[str, Any]], sizes: dict[str, int]) -> dict[str, Any]:
    g = ctx.data.games
    fin = g[g.status == "final"]
    season = cfg.current_season(ctx.today)
    seasons = sorted(g.season.unique())
    return {"generated_at": ctx.stamp, "season": season, "season_label": cfg.season_label(season),
            "latest_final_date": None if fin.empty else fin.date.max(),
            "counts": {"games": int(len(g)), "games_final": int(len(fin)), "teams": int(len(ctx.teams)),
                       "players": int(ctx.data.skaters.player_id.nunique()), "player_games": int(len(ctx.data.skaters)),
                       "props_rows": int(len(ctx.proj))},
            "model": {"game_version": ctx.gbundle["version"], "props_version": ctx.pbundle["version"], "trained_at": ctx.gbundle["trained_at"][:16] + "Z",
                      "train_games": ctx.gbundle["train_games"], "seasons": [cfg.season_label(int(s)) for s in seasons]},
            "schedule_days": [{"date": d["date"], "games": d["games"]} for d in days], "lineup_note": LINEUP_NOTE, "files": sizes}
