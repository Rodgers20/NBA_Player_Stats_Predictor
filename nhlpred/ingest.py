"""Pull the schedule and finished-game boxscores into parquet. Incremental: only unseen finished games are fetched."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

import pandas as pd

from nhlpred import config as cfg
from nhlpred import parse
from nhlpred.feed import MAX_WORKERS, FeedError, get_json
from nhlpred.store import read_json, read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)


def scan_schedule(season: int, today: date) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk /schedule week by week across a season (regular season only)."""
    cur = date(season - 1, *cfg.SEASON_OPEN)
    end = date(season, *cfg.SEASON_CLOSE)
    games: dict[int, dict[str, Any]] = {}
    teams: dict[int, dict[str, Any]] = {}
    seen: set[str] = set()
    while cur <= end and cur.isoformat() not in seen:
        seen.add(cur.isoformat())
        payload = get_json(f"{cfg.API}/schedule/{cur.isoformat()}")
        g, t = parse.schedule_games(payload or {})
        games.update({x["game_id"]: x for x in g})
        teams.update({x["id"]: x for x in t})
        nxt = (payload or {}).get("nextStartDate")
        if not nxt:
            break
        cur = date.fromisoformat(nxt)
    return list(games.values()), list(teams.values())


def _fetch_box(gid: int) -> tuple[int, dict[str, Any] | None]:
    try:
        return gid, get_json(f"{cfg.API}/gamecenter/{gid}/boxscore", missing_ok=True)
    except FeedError as exc:
        log.warning("boxscore %s failed: %s", gid, exc)
        return gid, None


def _with_shots(games: pd.DataFrame, old: pd.DataFrame, fresh: dict[int, dict[str, Any]], skaters: pd.DataFrame) -> pd.DataFrame:
    """Team shots on goal per game: new boxscores, else what was stored, else the sum of the skaters' shots.

    A rescanned schedule row carries no shots, so without the last two sources a re-ingest would wipe them.
    """
    games = games.copy()
    for col in ("home_sog", "away_sog"):
        if col not in games:
            games[col] = float("nan")
        if not old.empty and col in old:
            games[col] = games[col].fillna(games.game_id.map(old.set_index("game_id")[col]))
    for gid, s in fresh.items():
        for col, v in s.items():
            games.loc[games.game_id == gid, col] = v
    if not skaters.empty:
        tot = skaters.groupby(["game_id", "home"]).sog.sum().unstack()
        for col, side in (("home_sog", True), ("away_sog", False)):
            if side in tot:
                games[col] = games[col].fillna(games.game_id.map(tot[side]))
    return games.sort_values(["date", "start_utc", "game_id"]).reset_index(drop=True)


def ingest(seasons: list[int], today: date | None = None) -> dict[str, int]:
    today = today or date.today()
    old_games = read_parquet(cfg.games_path())
    sched, teams = [], []
    for s in seasons:
        g, t = scan_schedule(s, today)
        log.info("season %s: %s regular-season games on the schedule", s, len(g))
        sched += g
        teams += t
    games = pd.DataFrame(sched).drop_duplicates("game_id", keep="last")
    if not old_games.empty:                      # keep seasons we did not rescan
        games = pd.concat([old_games[~old_games.game_id.isin(games.game_id)], games], ignore_index=True)
    sk, gl = read_parquet(cfg.skaters_path()), read_parquet(cfg.goalies_path())
    have = set(sk.game_id) if not sk.empty else set()
    todo = games[(games.status == "final") & ~games.game_id.isin(have)].game_id.tolist()
    log.info("%s finished games to fetch", len(todo))
    shots: dict[int, dict[str, Any]] = {}
    new_sk, new_gl = [], []
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        for i, (gid, box) in enumerate(pool.map(_fetch_box, todo), 1):
            if box and box.get("playerByGameStats"):
                s, a, b = parse.boxscore_rows(box)
                shots[gid] = s
                new_sk += a
                new_gl += b
            if i % 500 == 0:
                log.info("  %s/%s boxscores", i, len(todo))
    if new_sk:
        sk = pd.concat([sk, pd.DataFrame(new_sk)], ignore_index=True)
        gl = pd.concat([gl, pd.DataFrame(new_gl)], ignore_index=True)
    games = _with_shots(games, old_games, shots, sk)
    write_parquet(games, cfg.games_path())
    if new_sk:
        write_parquet(sk, cfg.skaters_path())
        write_parquet(gl, cfg.goalies_path())
    old_teams = {t["id"]: t for t in read_json(cfg.teams_path(), [])}
    old_teams.update({t["id"]: t for t in teams})
    write_json(sorted(old_teams.values(), key=lambda t: t["abbr"]), cfg.teams_path(), compact=False)
    return {"games": len(games), "fetched": len(shots), "boxscores_missing": len(todo) - len(shots)}


def fetch_standings() -> int:
    """Current standings snapshot (the league's own table: points, streak, last 10)."""
    payload = get_json(f"{cfg.API}/standings/now")
    rows = (payload or {}).get("standings", [])
    if rows:
        write_json({"as_of": payload.get("standingsDateTimeUtc"), "standings": rows}, cfg.standings_path())
    return len(rows)


def refresh_current(today: date | None = None) -> dict[str, int]:
    """Daily job: rescan only the running season. Finished games already stored are skipped."""
    out = ingest([cfg.current_season(today)], today)
    try:
        out["standings"] = fetch_standings()
    except FeedError as exc:                       # a missing standings table must not stop the daily job
        log.warning("standings not refreshed: %s", exc)
    return out
