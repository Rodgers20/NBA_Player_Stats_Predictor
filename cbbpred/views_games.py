"""Schedule rows and full matchup payloads."""

from __future__ import annotations

import json
from typing import Any

from cbbpred import ratings
from cbbpred.context import Context

SIDES = ("home", "away")


def r(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)


def team_obj(ctx: Context, g: dict[str, Any], side: str) -> dict[str, Any]:
    tid, kickoff, season = g[f"{side}_id"], g["kickoff"], g["season"]
    conf_id = g[f"{side}_conf"] or ctx.team_conf.get(tid)
    team = ctx.teams.get(tid, {})
    return {
        "id": tid, "name": g[f"{side}_name"], "short": g[f"{side}_short"] or team.get("short"),
        "abbr": g[f"{side}_abbr"], "logo": ctx.logo(tid), "color": team.get("color"), "alt_color": team.get("alt_color"),
        "conference_id": conf_id, "conference": ctx.confs.get(conf_id or "", {}).get("short"),
        "rank": g[f"{side}_rank"], "d1": ctx.is_d1(tid),
        "record": ctx.logs.record(tid, kickoff, season),
        "conf_record": ctx.logs.record(tid, kickoff, season, conference=True) if conf_id in ctx.confs else None,
        "streak": ctx.logs.streak(tid, kickoff, season),
    }


def line_obj(ctx: Context, game_id: str) -> dict[str, Any] | None:
    ln = ctx.lines.get(game_id)
    if ln is None:
        return None
    flip = lambda v: None if v is None else -v  # noqa: E731 - stored home-favored-positive; shown as a betting spread
    return {"provider": ln["line_provider"], "spread_home": flip(ln["spread_line"]), "total": ln["total_line"],
            "home_ml": ln["home_moneyline"], "away_ml": ln["away_moneyline"],
            "open_spread_home": flip(ln.get("spread_open")), "open_total": ln.get("total_open"),
            "updated": None if ln["line_fetched_at"] is None else ln["line_fetched_at"].strftime("%Y-%m-%dT%H:%MZ")}


def _spread_text(abbr_home: str, abbr_away: str, p_cover: float | None, spread_line: float | None) -> str | None:
    if p_cover is None or spread_line is None:
        return None
    home_line = -spread_line
    return f"{abbr_home} {home_line:+g}" if p_cover >= 0.5 else f"{abbr_away} {-home_line:+g}"


def pick_obj(call: dict[str, Any] | None, g: dict[str, Any]) -> dict[str, Any] | None:
    if call is None:
        return None
    home_wins = call["p_home_win"] >= 0.5
    ats = None if call["p_home_cover"] is None else {
        "side": "home" if call["p_home_cover"] >= 0.5 else "away",
        "text": _spread_text(g["home_abbr"], g["away_abbr"], call["p_home_cover"], call["spread_line"]),
        "p_cover": r(max(call["p_home_cover"], 1 - call["p_home_cover"]), 3), "confidence": call["ats_confidence"]}
    total = None if call["p_over"] is None else {
        "side": "over" if call["p_over"] >= 0.5 else "under", "line": call["total_line"],
        "p": r(max(call["p_over"], 1 - call["p_over"]), 3), "confidence": call["total_confidence"]}
    return {
        "source": call["source"], "model_version": call["model_version"],
        "winner": "home" if home_wins else "away", "winner_abbr": g["home_abbr"] if home_wins else g["away_abbr"],
        "p_home": r(call["p_home_win"], 3), "p_away": r(call["p_away_win"], 3), "confidence": call["win_confidence"],
        "proj_margin": r(call["proj_margin"]), "proj_total": r(call["proj_total"]),
        "proj_home_pts": r(call["proj_home_pts"]), "proj_away_pts": r(call["proj_away_pts"]),
        "market_p_home": r(call["market_prob_home"], 3), "ats": ats, "total": total,
    }


def result_obj(ctx: Context, g: dict[str, Any]) -> dict[str, Any] | None:
    if g["status"] != "final" or g["home_score"] is None:
        return None
    gr = ctx.graded.get(g["game_id"], {})
    return {
        "home_score": g["home_score"], "away_score": g["away_score"], "overtimes": g["overtimes"],
        "margin": g["home_score"] - g["away_score"], "total": g["home_score"] + g["away_score"],
        "winner": "home" if g["home_score"] > g["away_score"] else "away",
        "winner_correct": gr.get("winner_correct"), "market_winner_correct": gr.get("market_winner_correct"),
        "ats_result": gr.get("ats_result"), "ats_correct": gr.get("ats_correct"),
        "total_result": gr.get("total_result"), "total_correct": gr.get("total_correct"),
        "margin_error": r(gr.get("margin_error")),
    }


def game_item(ctx: Context, g: dict[str, Any]) -> dict[str, Any]:
    """The compact row used by the schedule and as the header of a game file."""
    return {
        "id": g["game_id"], "date": g["gameday"], "time": g["gametime"],
        "kickoff": g["kickoff"].strftime("%Y-%m-%dT%H:%MZ"), "status": g["status"], "season": g["season"],
        "neutral": g["neutral"], "conf_game": g["conf_game"], "note": g["note"], "venue": g["venue"],
        "broadcast": g["broadcast"], "home": team_obj(ctx, g, "home"), "away": team_obj(ctx, g, "away"),
        "score": None if g["home_score"] is None else {"home": g["home_score"], "away": g["away_score"],
                                                       "overtimes": g["overtimes"]},
        "line": line_obj(ctx, g["game_id"]), "pick": pick_obj(ctx.calls.get(g["game_id"]), g),
        "result": result_obj(ctx, g),
    }


def _ratings_in(ctx: Context, g: dict[str, Any], side: str) -> dict[str, Any] | None:
    pre = ctx.pre.get(g["game_id"])
    if pre is None:
        return None
    base, off, deff = ctx.params.get("base", ratings.Params().base), pre[f"{side}_off"], pre[f"{side}_def"]
    upcoming = g["status"] != "final"
    return {"off": r(off), "def": r(deff), "power": r(off - deff), "off100": r(ratings.per100(off, base)),
            "def100": r(ratings.per100(deff, base)), "games_played": pre[f"{side}_gp"],
            "power_rank": ctx.power_rank.get(g[f"{side}_id"]) if upcoming else None}


def _factors(raw: str | None) -> list[dict[str, Any]]:
    return json.loads(raw) if raw else []


def _why(call: dict[str, Any] | None, g: dict[str, Any]) -> list[dict[str, Any]]:
    """The biggest margin factors as plain statements, each saying which team it favors."""
    if call is None:
        return []
    out = []
    for f in sorted(_factors(call.get("factors_margin")), key=lambda f: -abs(f["value"])):
        if abs(f["value"]) < 0.5:
            continue
        side = "home" if f["value"] > 0 else "away"
        out.append({"label": f["label"], "points": abs(f["value"]), "favors": side,
                    "abbr": g["home_abbr"] if side == "home" else g["away_abbr"]})
    return out


def game_detail(ctx: Context, g: dict[str, Any]) -> dict[str, Any]:
    item = game_item(ctx, g)
    call = ctx.calls.get(g["game_id"])
    k = g["kickoff"]
    form = {side: {"overall": ctx.logs.form(g[f"{side}_id"], k),
                   "split": ctx.logs.form(g[f"{side}_id"], k, at_home=(side == "home"))} for side in SIDES}
    h2h = ctx.logs.head_to_head(g["home_id"], g["away_id"], k)
    return {
        **item, "form": form, "h2h": h2h,
        "ratings": {side: _ratings_in(ctx, g, side) for side in SIDES},
        "line_movement": ctx.line_moves.get(g["game_id"], []),
        "factors": {"margin": _factors(call.get("factors_margin")) if call else [],
                    "total": _factors(call.get("factors_total")) if call else []},
        "why": _why(call, g),
    }
