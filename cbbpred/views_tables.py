"""Rankings, standings, model performance and health payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from cbbpred import config as cfg
from cbbpred import ratings
from cbbpred.context import Context
from cbbpred.form import FORM_GAMES
from cbbpred.grade import track_record, track_record_by_season
from cbbpred.store import read_json
from cbbpred.views_games import game_item, r

FAR_FUTURE = datetime.max.replace(tzinfo=UTC)


def current_season(ctx: Context) -> int:
    return int(ctx.games["season"].max())


def _team_slim(ctx: Context, tid: str) -> dict[str, Any]:
    t = ctx.teams.get(tid, {})
    conf = ctx.team_conf.get(tid)
    return {"id": tid, "name": t.get("name"), "short": t.get("short"), "abbr": t.get("abbr"), "logo": ctx.logo(tid),
            "color": t.get("color"), "conference_id": conf, "conference": ctx.confs.get(conf or "", {}).get("short")}


def rankings(ctx: Context) -> dict[str, Any]:
    """Power ratings of every Division I team, best first."""
    season = current_season(ctx)
    base = ctx.params.get("base", ratings.Params().base)
    ap = ctx.ap.get("ranks", {})
    rows = []
    for tid, tr in ctx.team_ratings.items():
        if not ctx.is_d1(tid):
            continue
        off100, def100 = ratings.per100(tr["off"], base), ratings.per100(tr["def"], base)
        rows.append({
            **_team_slim(ctx, tid), "power_rank": ctx.power_rank[tid], "ap_rank": (ap.get(tid) or {}).get("rank"),
            "ap_previous": (ap.get(tid) or {}).get("previous"),
            "record": ctx.logs.record(tid, FAR_FUTURE, season),
            "conf_record": ctx.logs.record(tid, FAR_FUTURE, season, conference=True),
            "last10": ctx.logs.form(tid, FAR_FUTURE)["record"], "games": tr["games"],
            "off": r(tr["off"]), "def": r(tr["def"]), "power": r(tr["power"]),
            "off100": r(off100), "def100": r(def100), "net100": r(off100 - def100)})
    rows.sort(key=lambda x: x["power_rank"])
    return {"generated_at": ctx.generated_at.strftime("%Y-%m-%dT%H:%MZ"), "season": season,
            "ap_week": ctx.ap.get("label"), "ap_updated": ctx.ap.get("updated"),
            "ratings_note": "off100/def100 are points per 100 possessions (points per game scaled by average pace); "
                            "off/def/power are points per game above an average Division I team.",
            "teams": rows}


def _standing_row(ctx: Context, tid: str, season: int, ap: dict[str, Any]) -> dict[str, Any]:
    past = [p for p in ctx.logs.before(tid, FAR_FUTURE) if p.season == season]
    conf = [p for p in past if p.conf_game]
    wins, cwins = sum(p.win for p in past), sum(p.win for p in conf)
    diff = sum(p.pts_for - p.pts_against for p in past)
    return {**_team_slim(ctx, tid), "ap_rank": (ap.get(tid) or {}).get("rank"),
            "conf_wins": cwins, "conf_losses": len(conf) - cwins, "conf_record": f"{cwins}-{len(conf) - cwins}",
            "wins": wins, "losses": len(past) - wins, "overall": f"{wins}-{len(past) - wins}",
            "diff": diff, "avg_diff": r(diff / len(past)) if past else None,
            "streak": ctx.logs.streak(tid, FAR_FUTURE, season)}


def standings(ctx: Context) -> dict[str, Any]:
    season, ap = current_season(ctx), ctx.ap.get("ranks", {})
    groups = []
    for cid, conf in sorted(ctx.confs.items(), key=lambda kv: kv[1]["name"]):
        teams = [_standing_row(ctx, tid, season, ap) for tid, c in ctx.team_conf.items() if c == cid]
        if not teams:
            continue
        teams.sort(key=lambda t: (-(t["conf_wins"] / max(1, t["conf_wins"] + t["conf_losses"])), -t["conf_wins"],
                                  -(t["wins"] / max(1, t["wins"] + t["losses"])), -t["diff"], t["name"] or ""))
        groups.append({"id": cid, "name": conf["name"], "short": conf["short"], "logo": conf.get("logo"), "teams": teams})
    return {"generated_at": ctx.generated_at.strftime("%Y-%m-%dT%H:%MZ"), "season": season, "conferences": groups}


def performance(ctx: Context, graded: pl.DataFrame) -> dict[str, Any]:
    path = cfg.model_dir() / "backtest_summary.json"
    bt = read_json(path) if path.exists() else {}
    live = graded.filter(pl.col("source") == "live") if graded.height else graded
    recent = [game_item(ctx, ctx.by_id[gid]) for gid in graded.tail(cfg.RECENT_GRADED)["game_id"].to_list()[::-1]] \
        if graded.height else []
    meta = ctx.model_meta
    return {
        "generated_at": ctx.generated_at.strftime("%Y-%m-%dT%H:%MZ"),
        "model": {"version": meta.get("version"), "trained_at": meta.get("trained_at"), "n_games": meta.get("n_games"),
                  "seasons": meta.get("seasons"), "rating_params": meta.get("rating_params")},
        "backtest": {k: bt.get(k) for k in ("n_games", "seasons", "model", "ratings_only", "home_team_always_wins",
                                            "sigma_margin", "sigma_total", "holdout")} if bt else None,
        "vs_market": bt.get("vs_market"), "by_season": bt.get("by_season", []),
        "calibration": bt.get("calibration", []),
        "track_record": {"all": track_record(graded), "live": track_record(live),
                         "by_season": track_record_by_season(graded)},
        "recent": recent,
        "notes": ["Backtest calls come from a model that only saw earlier seasons.",
                  "Sportsbook lines exist only for seasons ESPN kept them; see vs_market.seasons."],
    }


def health(ctx: Context, days: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    g = ctx.games
    finals = g.filter(pl.col("status") == "final")
    upcoming = g.filter((pl.col("status") == "scheduled") & (pl.col("kickoff") > ctx.generated_at)).sort("kickoff")
    return {
        "league": "cbb", "status": "ok", "generated_at": ctx.generated_at.strftime("%Y-%m-%dT%H:%MZ"),
        "season": current_season(ctx), "model_version": ctx.model_meta.get("version"),
        "games": {"total": g.height, "final": finals.height, "scheduled": upcoming.height},
        "latest_final_date": finals["gameday"].max() if finals.height else None,
        "next_game_date": upcoming["gameday"].min() if upcoming.height else None,
        "games_with_lines": len(ctx.lines), "ap_week": ctx.ap.get("label"), "form_games": FORM_GAMES,
        "schedule": {"from": days[0]["date"] if days else None, "to": days[-1]["date"] if days else None, "days": days},
        "files": counts,
    }
