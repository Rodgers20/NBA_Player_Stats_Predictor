"""Command line entry point: python -m cbbpred <command>."""

from __future__ import annotations

import argparse
import json
import logging

from cbbpred import config as cfg

log = logging.getLogger("cbbpred")


def cmd_ingest(args: argparse.Namespace) -> None:
    from cbbpred import lines
    from cbbpred.espn import current_season, ingest_games, refresh_reference
    season = current_season()
    seasons = [season] if args.current else list(range(cfg.FIRST_SEASON, season + 1))
    games = ingest_games(seasons, refresh=args.current)
    refresh_reference(season)
    print(f"{games.height} games on disk")
    if not args.no_lines:
        print("lines:", lines.update_lines(games))


def cmd_tune(_: argparse.Namespace) -> None:
    import polars as pl

    from cbbpred import ratings
    from cbbpred.model import d1_ids
    from cbbpred.store import write_json
    games = pl.read_parquet(cfg.games_path())
    last = int(games["season"].max())
    result = ratings.tune(games, d1_ids(), range(cfg.FIRST_SEASON, last - 1), range(last - 1, last + 1))
    write_json(result, cfg.model_dir() / "rating_params.json")
    print(json.dumps(result, indent=2))


def cmd_ratings(_: argparse.Namespace) -> None:
    from cbbpred.model import refresh_ratings
    print(f"{refresh_ratings().height} games rated")


def cmd_backtest(_: argparse.Namespace) -> None:
    from cbbpred.model import backtest
    s = backtest()
    keys = ("n_games", "model", "ratings_only", "home_team_always_wins", "vs_market")
    print(json.dumps({k: s[k] for k in keys}, indent=2, default=str))


def cmd_train(_: argparse.Namespace) -> None:
    from cbbpred.model import train
    print("production version:", train()["version"])


def cmd_predict(_: argparse.Namespace) -> None:
    from cbbpred.predict import predict_upcoming
    print(f"{predict_upcoming().height} predictions appended")


def cmd_grade(_: argparse.Namespace) -> None:
    from cbbpred.grade import grade, track_record
    print(json.dumps(track_record(grade()), indent=2, default=str))


def cmd_export(_: argparse.Namespace) -> None:
    from cbbpred.export import export_all
    print(json.dumps(export_all(), indent=2, default=str))


def _players(ctx):  # noqa: ANN001
    from cbbpred.espn import current_season
    from cbbpred.players import players_payload
    return players_payload(ctx, current_season())


def cmd_players(_: argparse.Namespace) -> None:
    from cbbpred.context import load
    from cbbpred.export import _write
    payload = _players(load())
    if not payload["players"]:
        print("no players returned; keeping the last players.json")
        return
    _write(cfg.export_dir() / "players.json", payload)
    print(f"{len(payload['players'])} players")


def cmd_draft(_: argparse.Namespace) -> None:
    """Draft board + pick ownership. A source page that stops parsing leaves the last good draft.json in place."""
    from cbbpred.context import load
    from cbbpred.draft import DraftFeedError, draft_payload
    from cbbpred.espn import current_season
    from cbbpred.export import _write
    ctx = load()
    players = _players(ctx)["players"]
    standings = cfg.export_dir().parent / "nba" / "standings.json"
    try:
        payload = draft_payload(players, standings, ctx.generated_at.strftime("%Y-%m-%dT%H:%MZ"), current_season())
    except DraftFeedError as exc:
        log.warning("draft board not refreshed: %s", exc)
        print("draft board not refreshed; keeping the last draft.json")
        return
    _write(cfg.export_dir() / "draft.json", payload)
    print(f"{len(payload['board'])} prospects, {sum(1 for b in payload['board'] if b['espn_id'])} linked to ESPN")


def cmd_refresh(args: argparse.Namespace) -> None:
    """Daily job: pull the running season, rebuild ratings, grade, predict what is ahead, write the JSON."""
    args.current, args.no_lines = True, False
    cmd_ingest(args)
    cmd_ratings(args)
    cmd_predict(args)
    cmd_grade(args)
    cmd_export(args)
    for extra in (cmd_players, cmd_draft):       # separate from export: they need the network and may fail alone
        try:
            extra(args)
        except Exception:  # noqa: BLE001 - one broken feed must not stop the daily job
            log.exception("%s failed; keeping the last file", extra.__name__)


COMMANDS = {
    "ingest": (cmd_ingest, "pull games, teams, conferences, AP poll and lines from ESPN (--current: running season only)"),
    "tune": (cmd_tune, "grid-search the rating update rule and report holdout error"),
    "ratings": (cmd_ratings, "rebuild pre-game ratings for every game"),
    "backtest": (cmd_backtest, "walk-forward backtest"),
    "train": (cmd_train, "fit the production model"),
    "predict": (cmd_predict, "project unplayed games and append changed ones to the log"),
    "grade": (cmd_grade, "grade pre-tip predictions against final scores"),
    "export": (cmd_export, "write the static JSON under frontend/public/data/cbb"),
    "players": (cmd_players, "write players.json: every D1 player with real minutes, this and last season"),
    "draft": (cmd_draft, "write draft.json: NBA draft board, pick ownership and lottery inputs"),
    "refresh": (cmd_refresh, "daily job: ingest --current, ratings, predict, grade, export"),
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="cbbpred")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (fn, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--current", action="store_true", help="only the running season (ingest)")
        p.add_argument("--no-lines", action="store_true", help="skip the sportsbook line fetch (ingest)")
        p.set_defaults(fn=fn)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parser.parse_args()
    args.fn(args)
