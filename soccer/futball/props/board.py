"""Assemble the `props/board` and `players/gamelog` payloads described in docs/props-contract.md."""
from __future__ import annotations

from typing import Callable

from . import forms as fm
from . import price

#: Rows below these probabilities are dropped: they are noise on a betting board and bloat the snapshot.
MIN_PROB = {("anytime_goal", None): 0.02, ("assist", None): 0.02, ("shots_on_target", 0.5): 0.05, ("shots_on_target", 1.5): 0.02}
NO_SOT_COMPETITIONS = ("MLS",)  # the match model has no shots history for these, and the contract promises no SOT props
NOTES = ["Tackles are not available from any free source",
         "Minutes are not published: appearances are weighted (start = 1, substitute = 0.3)"]
PERSON_FIELDS = ("player", "team", "opponent", "is_home", "position", "market", "line", "prob", "expected", "per_game",
                 "games", "start_prob", "confidence", "reason")


def _league_avgs(aggs: dict[str, fm.TeamAggregate], teams: set[str]) -> dict[str, float]:
    chosen = [a for t, a in aggs.items() if t in teams] or list(aggs.values())
    games = sum(a.games for a in chosen) or 1
    return {"goals": sum(a.goals_for for a in chosen) / games, "sot": sum(a.sot_for for a in chosen) / games}


def _team_expectations(prediction: dict) -> tuple[dict, dict | None]:
    goals = {"home": prediction["expected_goals"]["home"], "away": prediction["expected_goals"]["away"]}
    sot = prediction.get("sot") or {}
    shots = {"home": sot["home"]["expected"], "away": sot["away"]["expected"]} if sot.get("available") else None
    return goals, shots


def _keep(row: dict) -> bool:
    return row["prob"] >= MIN_PROB[(row["market"], row["line"])]


def build_board(code: str, rows: list[dict], fixtures: list[dict], predict: Callable[[str, str], dict],
                key_for: Callable[[str, str, str], str], generated_at: str, trained_through: str | None,
                dispersion: dict[str, float] | None = None, strength: dict[str, float] | None = None) -> dict:
    """Props for every fixture in `fixtures`. A fixture the match model cannot price is listed with no props."""
    forms = fm.build_forms(rows)
    aggs = fm.team_aggregates(rows)
    priors = price.position_priors(forms)
    scope = {f.team for f in forms if code == "UCL" or f.competition_code == code}
    league = _league_avgs(aggs, scope)
    markets = tuple(m for m in price.MARKETS if not (m == "shots_on_target" and code in NO_SOT_COMPETITIONS))

    out = []
    for fx in fixtures:
        entry = {k: fx[k] for k in ("fixture_key", "date", "time", "kickoff_utc", "home_team", "away_team")}
        try:
            xg, sot = _team_expectations(predict(fx["home_team"], fx["away_team"]))
        except (ValueError, KeyError, TypeError):
            out.append({**entry, "home_xg": None, "away_xg": None, "props": []})
            continue
        priced = price.price_fixture(forms, aggs, priors, league, fx["home_team"], fx["away_team"], xg, sot, markets, dispersion, strength)
        props = [{"player_key": key_for(r["espn_id"], r["player"], r["team"]), **{k: r[k] for k in PERSON_FIELDS}}
                 for r in priced if _keep(r)]
        out.append({**entry, "home_xg": xg["home"], "away_xg": xg["away"], "props": props})
    return {"competition_code": code, "generated_at": generated_at, "model_trained_through": trained_through, "fixtures": out}


def _season_label(code: str, latest: str) -> str:
    year, month = int(latest[:4]), int(latest[5:7])
    if code == "MLS":
        return str(year)
    start = year if month >= 7 else year - 1
    return f"{start}/{(start + 1) % 100:02d}"


def _result(team_goals: int, opp_goals: int) -> str:
    letter = "W" if team_goals > opp_goals else "L" if team_goals < opp_goals else "D"
    return f"{letter} {team_goals}-{opp_goals}"


STAT_COLUMNS = ("goals", "assists", "shots", "sot", "fouls_committed", "fouls_suffered", "yellow", "red", "offsides")


def gamelog(rows: list[dict], espn_id: str, player_key: str, board: dict) -> dict:
    """Season totals / per-game averages, every match with a box score, and his next priced fixture."""
    form = next(f for f in fm.build_forms(rows) if f.espn_id == espn_id)
    mine = sorted((r for r in rows if r["espn_id"] == espn_id and r["team"] == form.team),
                  key=lambda r: (r["date"], r["event_id"]), reverse=True)
    totals = {c: sum(r[c] for r in mine) for c in STAT_COLUMNS}
    per_game = {c: round(totals[c] / len(mine), 4) for c in ("goals", "assists", "shots", "sot", "fouls_committed", "fouls_suffered", "yellow")}
    next_fx, props = None, []
    for fx in board["fixtures"]:
        found = [p for p in fx["props"] if p["player_key"] == player_key]
        if found:
            side_home = fx["home_team"] == form.team
            next_fx = {"fixture_key": fx["fixture_key"], "opponent": fx["away_team"] if side_home else fx["home_team"],
                       "date": fx["date"], "is_home": side_home}
            props = found
            break
    return {
        "player_key": player_key, "player": form.player, "team": form.team, "competition_code": form.competition_code,
        "position": form.position, "espn_id": espn_id,
        "season": {"label": _season_label(form.competition_code, mine[0]["date"]), "games": len(mine), "starts": form.starts,
                   "totals": totals, "per_game": per_game},
        "games": [{"date": r["date"], "competition_code": r["competition_code"], "opponent": r["opponent"], "is_home": r["is_home"],
                   "result": _result(r["team_goals"], r["opp_goals"]), "started": r["started"],
                   **{c: r[c] for c in STAT_COLUMNS}} for r in mine],
        "next_fixture": next_fx, "props": props, "notes": list(NOTES),
    }
