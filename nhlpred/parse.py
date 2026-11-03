"""Pure parsers: NHL API JSON -> flat rows. No network here so they are cheap to test."""

from __future__ import annotations

from typing import Any

from nhlpred import config as cfg


def toi_minutes(text: str | None) -> float:
    """'11:16' -> 11.267; bad or missing -> 0."""
    try:
        m, s = str(text).split(":")
        return int(m) + int(s) / 60
    except (ValueError, AttributeError):
        return 0.0


def _name(d: dict[str, Any] | None) -> str:
    return (d or {}).get("default", "") or ""


def team_row(t: dict[str, Any]) -> dict[str, Any]:
    place, common = _name(t.get("placeName")), _name(t.get("commonName"))
    return {"id": int(t["id"]), "abbr": t["abbrev"], "name": f"{place} {common}".strip(), "short": common,
            "logo": t.get("logo"), "logo_dark": t.get("darkLogo")}


def status_of(state: str) -> str:
    if state in cfg.FINAL_STATES:
        return "final"
    if state in cfg.LIVE_STATES:
        return "live"
    return "scheduled"


def schedule_games(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Regular-season games (and the teams seen) from one /schedule/{date} week."""
    games, teams = [], {}
    for day in payload.get("gameWeek", []):
        for g in day.get("games", []):
            if g.get("gameType") != 2 or g.get("gameScheduleState") not in (None, "OK"):
                continue
            h, a = g["homeTeam"], g["awayTeam"]
            teams[h["id"]], teams[a["id"]] = team_row(h), team_row(a)
            final = status_of(g["gameState"]) == "final"
            games.append({
                "game_id": int(g["id"]), "season": int(str(g["season"])[4:]), "date": day["date"],
                "start_utc": g["startTimeUTC"], "state": g["gameState"], "status": status_of(g["gameState"]),
                "home_id": int(h["id"]), "away_id": int(a["id"]), "home_abbr": h["abbrev"], "away_abbr": a["abbrev"],
                "home_score": h.get("score") if final else None, "away_score": a.get("score") if final else None,
                "period_type": (g.get("gameOutcome") or {}).get("lastPeriodType") if final else None,
                "venue": _name(g.get("venue")),
            })
    return games, list(teams.values())


def boxscore_rows(box: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """(team shots by side, skater rows, goalie rows) from a finished game's boxscore."""
    gid, date, season = int(box["id"]), box["gameDate"], int(str(box["season"])[4:])
    sides = {"homeTeam": box["homeTeam"], "awayTeam": box["awayTeam"]}
    shots = {"home_sog": sides["homeTeam"].get("sog"), "away_sog": sides["awayTeam"].get("sog")}
    skaters, goalies = [], []
    for key, team in sides.items():
        opp = sides["awayTeam" if key == "homeTeam" else "homeTeam"]
        base = {"game_id": gid, "date": date, "season": season, "team": team["abbrev"], "team_id": int(team["id"]),
                "opp": opp["abbrev"], "home": key == "homeTeam"}
        stats = box.get("playerByGameStats", {}).get(key, {})
        for p in stats.get("forwards", []) + stats.get("defense", []):
            skaters.append({**base, "player_id": int(p["playerId"]), "name": _name(p.get("name")),
                            "pos": p.get("position"), "goals": int(p.get("goals", 0)), "assists": int(p.get("assists", 0)),
                            "points": int(p.get("points", 0)), "sog": int(p.get("sog", 0)),
                            "pp_goals": int(p.get("powerPlayGoals", 0)), "toi": toi_minutes(p.get("toi")),
                            "toi_text": p.get("toi", "0:00"), "hits": int(p.get("hits", 0)),
                            "blocks": int(p.get("blockedShots", 0)), "shifts": int(p.get("shifts", 0))})
        for p in stats.get("goalies", []):
            toi = toi_minutes(p.get("toi"))
            if toi <= 0:
                continue
            goalies.append({**base, "player_id": int(p["playerId"]), "name": _name(p.get("name")),
                            "starter": bool(p.get("starter")), "shots_against": int(p.get("shotsAgainst", 0)),
                            "saves": int(p.get("saves", 0)), "goals_against": int(p.get("goalsAgainst", 0)),
                            "toi": toi, "toi_text": p.get("toi", "0:00"), "decision": p.get("decision")})
    return shots, skaters, goalies
