"""Props board, per-game top props, players index and player pages."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nhlpred import config as cfg
from nhlpred import prop_model as pm
from nhlpred.context import Context, nan_to_none
from nhlpred.oddsmath import fair_american

LINE_P_MIN, LINE_P_MAX = 0.03, 0.97
MIN_HIT_GAMES = 5
TOP_PROPS_PER_GAME, TOP_PER_MARKET = 8, 3
LOG_GAMES = 50
SOG_PICK_P = 0.55
LINEUP_NOTE = ("Projected lineup: players who dressed in their team's recent games. The free NHL API has no scratch or injury list, "
               "so confirm lineups before betting.")


def headshot(team: str, pid: int, season: int) -> str:
    return cfg.MUG.format(season8=f"{season - 1}{season}", team=team, pid=pid)


def _history(ctx: Context) -> dict[int, dict[str, np.ndarray]]:
    """Per player: arrays of every finished game (oldest first) for the stats and flags the exports use."""
    if "hist" not in ctx.cache:
        out = {}
        cols = ["goals", "assists", "points", "sog", "toi", "pp_goals"]
        for pid, g in ctx.players.sort_values(["start_utc", "game_id"]).groupby("player_id"):
            out[int(pid)] = {**{c: g[c].values.astype(float) for c in cols}, "season": g.season.values, "home": g.home.values, "date": g.date.values}
        ctx.cache["hist"] = out
    return ctx.cache["hist"]


def _hit(arr: np.ndarray, line: float, n: int | None) -> float | None:
    w = arr if n is None else arr[-n:]
    return None if len(w) < MIN_HIT_GAMES else round(float((w > line).mean()), 3)


def _market(ctx: Context, i: int, market: str, hist: dict[str, np.ndarray] | None, season: int) -> dict[str, Any]:
    m = pm.MARKETS[market]
    arr = hist[m["target"]] if hist else np.array([])
    cur = arr[hist["season"] == season] if hist else arr
    lines = []
    for ln in m["lines"]:
        p = float(ctx.pprob[market]["p"][ln][i])
        if market != "goal" and not LINE_P_MIN < p < LINE_P_MAX:
            continue
        lines.append({"line": ln, "p_over": round(p, 4), "p_under": round(1 - p, 4), "fair_over": fair_american(p), "fair_under": fair_american(1 - p),
                      "hit_l10": _hit(arr, ln, 10), "hit_l20": _hit(arr, ln, 20), "hit_season": _hit(cur, ln, None)})
    if not lines:                                        # keep the line closest to a coin flip so the market is never empty
        ps = {ln: float(ctx.pprob[market]["p"][ln][i]) for ln in m["lines"]}
        ln = min(ps, key=lambda k: abs(ps[k] - 0.5))
        lines.append({"line": ln, "p_over": round(ps[ln], 4), "p_under": round(1 - ps[ln], 4), "fair_over": fair_american(ps[ln]),
                      "fair_under": fair_american(1 - ps[ln]), "hit_l10": _hit(arr, ln, 10), "hit_l20": _hit(arr, ln, 20), "hit_season": _hit(cur, ln, None)})
    return {"projection": round(float(ctx.pprob[market]["mu"][i]), 3),
            "season_avg": round(float(cur.mean()), 3) if len(cur) >= 3 else None,
            "l10_avg": round(float(arr[-10:].mean()), 3) if len(arr) >= MIN_HIT_GAMES else None,
            "confidence": pm.confidence(float(ctx.proj.gp_l40.iloc[i])), "lines": lines}


def _row(ctx: Context, i: int) -> dict[str, Any]:
    r = ctx.proj.iloc[i]
    pid = int(r.player_id)
    hist = _history(ctx).get(pid)
    season = int(r.season)
    return {"player_id": pid, "name": r["name"], "team": r.team, "pos": r.pos, "headshot": headshot(r.team, pid, season), "opp": r.opp,
            "home": bool(r.home), "game_id": int(r.game_id), "date": r.date.strftime("%Y-%m-%d"),
            "start": pd.Timestamp(r.start_utc).strftime("%Y-%m-%dT%H:%MZ"), "lineup": "projected", "proj_toi": round(float(r.toi_l10), 1),
            "markets": {mk: _market(ctx, i, mk, hist, season) for mk in pm.MARKETS}}


def _board_idx(ctx: Context) -> list[int]:
    if ctx.proj.empty:
        return []
    days = sorted(ctx.proj.date.dt.strftime("%Y-%m-%d").unique())
    days = [d for d in days if d >= ctx.today.isoformat()][: cfg.PROPS_DAYS_AHEAD + 1]
    return [i for i in range(len(ctx.proj)) if ctx.proj.date.iloc[i].strftime("%Y-%m-%d") in days]


def props_board(ctx: Context) -> dict[str, Any]:
    idx = _board_idx(ctx)
    rows = [_row(ctx, i) for i in idx]
    rows.sort(key=lambda r: (r["date"], r["start"], -r["markets"]["goal"]["lines"][0]["p_over"]))
    return {"generated_at": ctx.stamp, "dates": sorted({r["date"] for r in rows}), "lineup_note": LINEUP_NOTE,
            "model": {"version": ctx.pbundle["version"]}, "players": rows}


def _base_rates() -> dict[tuple[str, float], float]:
    s = pm.summary_json().get("backtest", {})
    return {(mk, ln["line"]): ln["hit_rate"] for mk, v in s.items() for ln in v.get("lines", [])}


def top_props_for_game(ctx: Context, game_id: int) -> list[dict[str, Any]]:
    """Best props by lift over the league hit rate for that line: a 40% goal chance beats a 70% 'over 0.5 shots'."""
    base = _base_rates()
    picks = []
    for i in np.flatnonzero(ctx.proj.game_id.values == game_id):
        r = ctx.proj.iloc[i]
        for mk, m in pm.MARKETS.items():
            ln = m["main"]
            if mk == "sog":
                ok = [x for x in m["lines"] if float(ctx.pprob[mk]["p"][x][i]) >= SOG_PICK_P]
                if not ok:
                    continue
                ln = max(ok)
            p = float(ctx.pprob[mk]["p"][ln][i])
            b = base.get((mk, ln))
            if not b:
                continue
            picks.append({"player_id": int(r.player_id), "name": r["name"], "team": r.team, "pos": r.pos,
                          "headshot": headshot(r.team, int(r.player_id), int(r.season)), "market": mk, "line": ln, "side": "over", "p": round(p, 4),
                          "fair_odds": fair_american(p), "projection": round(float(ctx.pprob[mk]["mu"][i]), 3),
                          "confidence": pm.confidence(float(r.gp_l40)), "lift": round(p / b, 2)})
    picks.sort(key=lambda x: -x["lift"])
    out, per = [], {}
    for p in picks:
        if per.get(p["market"], 0) < TOP_PER_MARKET and p["confidence"] != "low":
            out.append(p)
            per[p["market"]] = per.get(p["market"], 0) + 1
        if len(out) >= TOP_PROPS_PER_GAME:
            break
    return out


def _season_lines(ctx: Context) -> pd.DataFrame:
    """Season totals for the current season per player."""
    season = cfg.current_season(ctx.today)
    cur = ctx.players[ctx.players.season == season]
    return cur.groupby("player_id").agg(gp=("game_id", "count"), goals=("goals", "sum"), assists=("assists", "sum"), points=("points", "sum"),
                                         sog=("sog", "sum"), toi=("toi", "mean"), pp_goals=("pp_goals", "sum"))


def players_index(ctx: Context) -> dict[str, Any]:
    season = cfg.current_season(ctx.today)
    sk = ctx.players[ctx.players.season >= season - 1]
    gp = sk.groupby("player_id").game_id.count()
    keep = gp[gp >= 5].index
    latest = sk[sk.player_id.isin(keep)].sort_values(["start_utc", "game_id"]).drop_duplicates("player_id", keep="last").set_index("player_id")
    cur = _season_lines(ctx).reindex(latest.index).fillna(0)
    rows = [{"id": int(pid), "name": r["name"], "team": r.team, "pos": r.pos, "headshot": headshot(r.team, int(pid), season),
             "gp": int(cur.loc[pid, "gp"]), "goals": int(cur.loc[pid, "goals"]), "assists": int(cur.loc[pid, "assists"]),
             "points": int(cur.loc[pid, "points"]), "sog": int(cur.loc[pid, "sog"]), "toi": round(float(cur.loc[pid, "toi"]), 1)}
            for pid, r in latest.iterrows()]
    rows.sort(key=lambda r: (-r["points"], r["name"]))
    return {"generated_at": ctx.stamp, "season": season, "players": rows}


def _avg_block(h: dict[str, np.ndarray], n: int | None, mask: np.ndarray | None = None) -> dict[str, Any] | None:
    sel = slice(None) if n is None else slice(-n, None)
    idx = np.arange(len(h["goals"]))[mask] if mask is not None else np.arange(len(h["goals"]))
    idx = idx[sel]
    if len(idx) == 0:
        return None
    return {"gp": int(len(idx)), **{k: round(float(h[k][idx].mean()), 2) for k in ("goals", "assists", "points", "sog", "toi")}}


def _team_ranks(ctx: Context) -> dict[int, dict[str, int]]:
    """Rank of each team by shots / goals allowed (1 = allows the most), from rolling pre-game values at its next game."""
    if "ranks" not in ctx.cache:
        p = ctx.data.panel
        nxt = p[p.gf.isna()].sort_values("start_utc").drop_duplicates("team_id")
        if nxt.empty:
            nxt = p.sort_values("start_utc").drop_duplicates("team_id", keep="last")
        sa, ga = nxt.set_index("team_id").sa30.rank(ascending=False, method="min"), nxt.set_index("team_id").ga30.rank(ascending=False, method="min")
        ctx.cache["ranks"] = {int(t): {"rank_sa": int(sa[t]), "rank_ga": int(ga[t])} for t in sa.index}
    return ctx.cache["ranks"]


def _opp_context(ctx: Context, i: int) -> dict[str, Any]:
    r = ctx.proj.iloc[i]
    f = ctx.frame.loc[r.game_id]
    side = "a" if r.home else "h"
    gid = nan_to_none(f.get(f"goalie_id_{side}"))
    return {"sa_pg": round(float(r.opp_sa10), 1), "ga_pg": round(float(r.opp_ga10), 2), "opp_goalie": ctx.goalie_names.get(int(gid)) if gid is not None else None,
            "goalie_sv_pct": round(float(r.opp_svp), 3), **_team_ranks(ctx).get(int(r.opp_id), {})}


def player_page(ctx: Context, pid: int, meta: dict[str, Any], nxt_idx: int | None) -> dict[str, Any]:
    season = cfg.current_season(ctx.today)
    h = _history(ctx)[pid]
    in_cur, in_two = h["season"] == season, h["season"] >= season - 1
    cur = _avg_block(h, None, in_cur) or {"gp": 0}
    pp = float(h["pp_goals"][in_cur].sum())
    shots = float(h["sog"][in_cur].sum())
    season_block = {"gp": cur["gp"], "goals": int(h["goals"][in_cur].sum()), "assists": int(h["assists"][in_cur].sum()),
                    "points": int(h["points"][in_cur].sum()), "sog": int(shots), "toi": cur.get("toi"), "pp_goals": int(pp),
                    "shooting_pct": round(float(h["goals"][in_cur].sum() / shots), 3) if shots else None}
    rolling = {"l5": _avg_block(h, 5), "l10": _avg_block(h, 10), "l20": _avg_block(h, 20), "season": _avg_block(h, None, in_cur)}
    hit_rates = {}
    for mk, m in pm.MARKETS.items():
        arr = h[m["target"]]
        hit_rates[mk] = [{"line": ln, "l5": _hit(arr, ln, 5), "l10": _hit(arr, ln, 10), "l20": _hit(arr, ln, 20), "season": _hit(arr[in_cur], ln, None)}
                         for ln in m["lines"]]
    g = ctx.players[(ctx.players.player_id == pid) & (ctx.players.season >= season - 1)].sort_values(["start_utc", "game_id"]).tail(LOG_GAMES)
    games = ctx.data.games.set_index("game_id")
    log = []
    for r in g.iloc[::-1].itertuples():
        gr = games.loc[r.game_id]
        mine, theirs = (gr.home_score, gr.away_score) if r.home else (gr.away_score, gr.home_score)
        log.append({"game_id": int(r.game_id), "date": r.date.strftime("%Y-%m-%d"), "opp": r.opp, "home": bool(r.home), "toi": r.toi_text, "goals": int(r.goals),
                    "assists": int(r.assists), "points": int(r.points), "sog": int(r.sog), "pp_goals": int(r.pp_goals),
                    "result": "W" if mine > theirs else ("L" if gr.period_type == "REG" else "OTL")})
    two = {k: v[in_two] for k, v in h.items()}
    splits = {"home": _avg_block(two, None, two["home"].astype(bool)), "away": _avg_block(two, None, ~two["home"].astype(bool))}
    nxt = None
    if nxt_idx is not None:
        r = ctx.proj.iloc[nxt_idx]
        nxt = {"game_id": int(r.game_id), "date": r.date.strftime("%Y-%m-%d"), "opp": r.opp, "home": bool(r.home), "lineup": "projected",
               "markets": {mk: _market(ctx, nxt_idx, mk, h, int(r.season)) for mk in pm.MARKETS}, "opp_context": _opp_context(ctx, nxt_idx)}
    return {"player": {"id": pid, "name": meta["name"], "team": meta["team"], "pos": meta["pos"], "headshot": meta["headshot"]},
            "generated_at": ctx.stamp, "season": season_block, "rolling": rolling, "hit_rates": hit_rates, "log": log, "next": nxt, "splits": splits}


def next_game_index(ctx: Context) -> dict[int, int]:
    """player -> row index in ``ctx.proj`` of their next projected game."""
    if ctx.proj.empty:
        return {}
    first = ctx.proj.reset_index().sort_values(["start_utc", "game_id"]).drop_duplicates("player_id")
    return {int(r.player_id): int(r["index"]) for _, r in first.iterrows()}
