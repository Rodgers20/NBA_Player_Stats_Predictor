"""Glue between the lake and the pure props code: fixtures, the match predictor, player ids, search hits."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable

import polars as pl

from . import board as board_mod
from . import model

POSITION_NAMES = {"G": "Goalkeeper", "D": "Defence", "M": "Midfield", "F": "Offence"}  # the vocabulary the profile card uses
BOARD_DAYS = 14


def upcoming_fixtures(fixtures: pl.DataFrame, code: str, today: date | None = None, days: int = BOARD_DAYS) -> list[dict]:
    """Scheduled fixtures of one competition from today through `days` ahead, soonest first."""
    today = today or date.today()
    lo, hi = today.isoformat(), (today + timedelta(days=days)).isoformat()
    sub = fixtures.filter((pl.col("competition_code") == code) & (pl.col("status") == "scheduled")
                          & (pl.col("date") >= lo) & (pl.col("date") <= hi)).sort(["date", "time"])
    return sub.to_dicts()


def make_key_resolver(identities: dict[str, dict], lake_name: Callable[[str, str], str | None],
                      public_id: Callable[[dict], str]) -> Callable[[str, str, str], str]:
    """(espn_id, player, lake team) -> the existing public profile id when name AND club match, else `espn-<id>`."""
    from ..players import _match_key

    known: dict[tuple[str, str], str] = {}
    for entry in identities.values():
        for row in entry["league"]:
            club = lake_name(row.get("team"), row.get("competition_code"))
            if club:
                known[(_match_key(entry["name"]), club)] = public_id(entry)

    def key_for(espn_id: str, player: str, team: str) -> str:
        return known.get((_match_key(player), team)) or f"espn-{espn_id}"

    return key_for


def default_key_resolver() -> Callable[[str, str, str], str]:
    from .. import league, players

    return make_key_resolver(players._identities(), league._to_lake_name, players.public_id)


def default_predictor() -> Callable[[str, str], dict]:
    from ..predict import serve

    def predict(code: str):
        return lambda home, away: serve.predict(code, home, away)

    return predict


def build_boards(codes: list[str], stats_rows: list[dict], fixtures: pl.DataFrame, predict_for: Callable[[str], Callable],
                 key_for: Callable[[str, str, str], str], today: date | None = None) -> dict[str, dict]:
    """One props board per competition that has box scores and upcoming fixtures."""
    params = model.load_params()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    boards: dict[str, dict] = {}
    for code in codes:
        fx = upcoming_fixtures(fixtures, code, today)
        predict = predict_for(code)
        trained = None
        if fx:
            try:
                trained = predict(fx[0]["home_team"], fx[0]["away_team"])["context"]["model_trained_through"]
            except (ValueError, KeyError, TypeError):
                trained = None
        boards[code] = board_mod.build_board(code, stats_rows, fx, predict, key_for, now, trained, params["dispersion"], params["strength"])
    return boards


def key_index(stats_rows: list[dict], boards: dict[str, dict], key_for: Callable[[str, str, str], str]) -> dict[str, str]:
    """player_key -> espn_id for every player on any board."""
    wanted = {p["player_key"] for b in boards.values() for fx in b["fixtures"] for p in fx["props"]}
    out: dict[str, str] = {}
    for r in stats_rows:
        key = key_for(r["espn_id"], r["player"], r["team"])
        if key in wanted:
            out[key] = r["espn_id"]
    return out


def espn_search_hits(logs: dict[str, dict], colors: Callable[[str | None], dict], competition_names: dict[str, str]) -> list[dict]:
    """Search-index rows for players who only exist in the box scores (no football-data profile)."""
    from ..teams import normalise

    hits = []
    for key, log in logs.items():
        if not key.startswith("espn-"):
            continue
        hits.append({
            "player_id": key, "name": log["player"], "club": log["team"], "crest": None, "colors": colors(log["team"]),
            "nationality": None, "age": None, "position": POSITION_NAMES.get(log["position"] or ""),
            "competitions": [competition_names.get(log["competition_code"], log["competition_code"])],
            "has_full_stats": False, "goals": log["season"]["totals"]["goals"], "norm": normalise(log["player"]),
        })
    return hits
