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


def cmd_refresh(args: argparse.Namespace) -> None:
    """Daily job: pull the running season, rebuild ratings, grade, predict what is ahead, write the JSON."""
    args.current, args.no_lines = True, False
    cmd_ingest(args)
    cmd_ratings(args)
    cmd_predict(args)
    cmd_grade(args)
    cmd_export(args)


COMMANDS = {
    "ingest": (cmd_ingest, "pull games, teams, conferences, AP poll and lines from ESPN (--current: running season only)"),
    "tune": (cmd_tune, "grid-search the rating update rule and report holdout error"),
    "ratings": (cmd_ratings, "rebuild pre-game ratings for every game"),
    "backtest": (cmd_backtest, "walk-forward backtest"),
    "train": (cmd_train, "fit the production model"),
    "predict": (cmd_predict, "project unplayed games and append changed ones to the log"),
    "grade": (cmd_grade, "grade pre-tip predictions against final scores"),
    "export": (cmd_export, "write the static JSON under frontend/public/data/cbb"),
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
