"""Schedule items and full matchup pages."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nhlpred import game_model as gm
from nhlpred.context import Context, confidence, nan_to_none, team_ref
from nhlpred.lines import ev_per_unit, no_vig_home
from nhlpred.oddsmath import fair_american
from nhlpred.views_props import top_props_for_game

GROUPS = {"elo": ["f_elo"], "goal_diff": ["f_gd10", "f_gd30"], "shots": ["f_sd10", "f_sd30"], "form": ["f_win10"],
          "goalie": ["f_goalie"], "rest": ["rest_h", "rest_a", "b2b_h", "b2b_a"]}
GROUP_LABELS = {"elo": "Team strength (Elo)", "goal_diff": "Goal differential", "shots": "Shot differential", "form": "Recent form",
                "goalie": "Starting goalie", "rest": "Rest / schedule", "home_ice": "Home ice"}


def _goalie(ctx: Context, row: pd.Series, side: str) -> dict[str, Any]:
    gid = nan_to_none(row.get(f"goalie_id_{side}"))
    if gid is None:
        return {"id": None, "name": None, "sv_pct": None, "projected": True}
    return {"id": int(gid), "name": ctx.goalie_names.get(int(gid)), "sv_pct": round(float(row[f"svp_{side}"]), 3),
            "projected": bool(row.get(f"goalie_projected_{side}", True))}


def _pick(ctx: Context, g: pd.Series) -> dict[str, Any] | None:
    if g.game_id not in ctx.gpred.index:
        return None
    p = ctx.gpred.loc[g.game_id]
    ph = float(p.p_home)
    winner = "home" if ph >= 0.5 else "away"
    correct = None
    if g.status == "final":
        correct = (g.home_score > g.away_score) == (winner == "home")
    return {"source": p.source, "winner": winner, "winner_abbr": g.home_abbr if winner == "home" else g.away_abbr,
            "p_home": round(ph, 4), "p_away": round(1 - ph, 4), "confidence": confidence(ph), "proj_total": round(float(p.proj_total), 2),
            "proj_home": round(float(p.proj_home), 2), "proj_away": round(float(p.proj_away), 2), "correct": None if correct is None else bool(correct)}


LINE_KEYS = ("book", "home_ml", "away_ml", "total", "over_odds", "under_odds", "puck_home", "home_spread_odds", "away_spread_odds", "fetched_at", "open")


def _lines(ctx: Context, g: pd.Series, pick: dict[str, Any] | None) -> dict[str, Any] | None:
    """ESPN's sportsbook line for the game, plus how the model's win chance compares with the no-vig market."""
    row = ctx.lines.get(str(int(g.game_id)))
    if not row:
        return None
    out = {k: row.get(k) for k in LINE_KEYS}
    mp = no_vig_home(row)
    out["market_p_home"] = None if mp is None else round(mp, 4)
    out["ev_home"] = out["ev_away"] = None
    if pick and mp is not None:
        eh, ea = ev_per_unit(pick["p_home"], row.get("home_ml")), ev_per_unit(pick["p_away"], row.get("away_ml"))
        out["ev_home"], out["ev_away"] = (None if eh is None else round(eh, 4)), (None if ea is None else round(ea, 4))
    return out


def game_item(ctx: Context, g: pd.Series, counts: dict[int, int]) -> dict[str, Any]:
    f = ctx.frame.loc[g.game_id]
    final = g.status == "final"
    pick = _pick(ctx, g)

    def side(tid: int, abbr: str, score: Any) -> dict[str, Any]:
        return {**team_ref(ctx, tid), "record": _record(ctx, abbr), "score": None if not final else int(score)}

    return {"id": int(g.game_id), "date": g.date, "start": pd.Timestamp(g.start_utc).strftime("%Y-%m-%dT%H:%MZ"), "status": g.status,
            "game_type": "regular", "venue": g.venue,
            "home": side(g.home_id, g.home_abbr, g.home_score), "away": side(g.away_id, g.away_abbr, g.away_score),
            "period_type": g.period_type if final else None, "pick": pick, "lines": _lines(ctx, g, pick),
            "goalies": {"home": _goalie(ctx, f, "h"), "away": _goalie(ctx, f, "a")}, "props_count": int(counts.get(int(g.game_id), 0))}


def _record(ctx: Context, abbr: str) -> str | None:
    for r in ctx.standings:
        if r["teamAbbrev"]["default"] == abbr:
            return f"{r['wins']}-{r['losses']}-{r['otLosses']}"
    return None


def _form(ctx: Context, team_id: int, before: str, start: str) -> dict[str, Any]:
    p = ctx.data.panel
    t = p[(p.team_id == team_id) & p.gf.notna() & (p.start_utc < start)].tail(10)
    games = ctx.data.games.set_index("game_id")
    seq = []
    for r in t.itertuples():
        if r.win == 1:
            seq.append("W")
        else:
            seq.append("L" if games.loc[r.game_id].period_type == "REG" else "OTL")
    w, otl = seq.count("W"), seq.count("OTL")
    n = max(len(t), 1)
    return {"last10": seq[::-1], "record_l10": f"{w}-{len(seq) - w - otl}-{otl}", "gf": round(float(t.gf.mean()), 2) if len(t) else None,
            "ga": round(float(t.ga.mean()), 2) if len(t) else None, "sf": round(float(t.sf.mean()), 1) if len(t) else None,
            "sa": round(float(t.sa.mean()), 1) if len(t) else None, "win_pct": round(w / n, 3) if len(t) else None}


def _h2h(ctx: Context, g: pd.Series) -> dict[str, Any]:
    gm_ = ctx.data.games
    m = gm_[(gm_.status == "final") & (gm_.start_utc < g.start_utc) &
            (((gm_.home_id == g.home_id) & (gm_.away_id == g.away_id)) | ((gm_.home_id == g.away_id) & (gm_.away_id == g.home_id)))].tail(5)
    hw = sum(1 for r in m.itertuples() if (r.home_id == g.home_id) == (r.home_score > r.away_score))
    rows = [{"date": r.date, "home": r.home_abbr, "away": r.away_abbr, "home_score": int(r.home_score), "away_score": int(r.away_score)}
            for r in m.itertuples()][::-1]
    return {"home_wins": hw, "away_wins": len(m) - hw, "games": rows}


def _factors(ctx: Context, g: pd.Series) -> list[dict[str, Any]]:
    f = ctx.frame.loc[g.game_id]
    contrib = {c["key"]: c["pct"] for c in gm.explain(ctx.gbundle["models"], f)}
    vals = {
        "elo": (round(float(f.elo_h)), round(float(f.elo_a))),
        "goal_diff": (f"{f.gf10_h - f.ga10_h:+.2f}/g", f"{f.gf10_a - f.ga10_a:+.2f}/g"),
        "shots": (f"{f.sf10_h - f.sa10_h:+.1f}/g", f"{f.sf10_a - f.sa10_a:+.1f}/g"),
        "form": (f"{f.win10_h:.0%}", f"{f.win10_a:.0%}"),
        "goalie": (f"{f.svp_h:.3f}", f"{f.svp_a:.3f}"),
        "rest": ("B2B" if f.b2b_h else f"{int(f.rest_h)} days", "B2B" if f.b2b_a else f"{int(f.rest_a)} days")}
    out = []
    for key, feats in GROUPS.items():
        pct = sum(contrib[x] for x in feats)
        out.append({"key": key, "label": GROUP_LABELS[key], "home": vals[key][0], "away": vals[key][1],
                    "edge": "even" if abs(pct) < 0.3 else "home" if pct > 0 else "away", "pct": round(pct, 2)})
    lr = ctx.gbundle["models"]["win"].steps[1][1]
    ice = float((1 / (1 + np.exp(-lr.intercept_[0])) - 0.5) * 100)
    out.append({"key": "home_ice", "label": GROUP_LABELS["home_ice"], "home": "home", "away": "away", "edge": "home", "pct": round(ice, 2)})
    return sorted(out, key=lambda x: -abs(x["pct"]))


def _box(ctx: Context, gid: int, g: pd.Series) -> dict[str, Any]:
    sk = ctx.data.skaters[ctx.data.skaters.game_id == gid]
    gl = ctx.data.goalies[ctx.data.goalies.game_id == gid]

    def skaters(home: bool) -> list[dict[str, Any]]:
        d = sk[sk.home == home].sort_values(["points", "goals", "toi"], ascending=False)
        return [{"player_id": int(r.player_id), "name": r.name, "pos": r.pos, "toi": r.toi_text, "goals": int(r.goals), "assists": int(r.assists),
                 "points": int(r.points), "sog": int(r.sog), "pp_goals": int(r.pp_goals)} for r in d.itertuples()]

    def goalies(home: bool) -> list[dict[str, Any]]:
        d = gl[gl.home == home]
        return [{"player_id": int(r.player_id), "name": r.name, "saves": int(r.saves), "shots_against": int(r.shots_against),
                 "sv_pct": round(r.saves / r.shots_against, 3) if r.shots_against else None, "toi": r.toi_text,
                 "decision": nan_to_none(r.decision), "starter": bool(r.starter)} for r in d.itertuples()]

    return {"sog": {"home": nan_to_none(g.get("home_sog")), "away": nan_to_none(g.get("away_sog"))},
            "skaters": {"home": skaters(True), "away": skaters(False)}, "goalies": {"home": goalies(True), "away": goalies(False)}}


def game_detail(ctx: Context, g: pd.Series, counts: dict[int, int]) -> dict[str, Any]:
    f = ctx.frame.loc[g.game_id]
    item = game_item(ctx, g, counts)
    return {"game": item,
            "form": {"home": _form(ctx, int(g.home_id), g.date, g.start_utc), "away": _form(ctx, int(g.away_id), g.date, g.start_utc)},
            "rest": {"home": {"days": int(f.rest_h), "b2b": bool(f.b2b_h)}, "away": {"days": int(f.rest_a), "b2b": bool(f.b2b_a)}},
            "elo": {"home": round(float(f.elo_h)), "away": round(float(f.elo_a))}, "h2h": _h2h(ctx, g), "factors": _factors(ctx, g),
            "top_props": top_props_for_game(ctx, int(g.game_id)), "box": _box(ctx, int(g.game_id), g) if g.status == "final" else None,
            "fair_odds": {"home": fair_american(item["pick"]["p_home"]) if item["pick"] else None,
                          "away": fair_american(item["pick"]["p_away"]) if item["pick"] else None}}
