"""Command line entry point: python -m nhlpred <command>."""

from __future__ import annotations

import argparse
import json
import logging

from nhlpred import config as cfg

log = logging.getLogger("nhlpred")


def cmd_ingest(args: argparse.Namespace) -> None:
    from nhlpred.ingest import fetch_standings, ingest, refresh_current
    if args.current:
        print(refresh_current())
    else:
        print(ingest(list(range(cfg.FIRST_SEASON, cfg.current_season() + 1))))
        print("standings:", fetch_standings())


def cmd_train(_: argparse.Namespace) -> None:
    from nhlpred.pipeline import train_all
    print(json.dumps(train_all(), indent=2))


def cmd_backtest(_: argparse.Namespace) -> None:
    """Walk-forward backtests for both model families (also refits and saves the production models)."""
    from nhlpred import game_model as gm
    from nhlpred import prop_model as pm
    from nhlpred.pipeline import train_all
    train_all()
    g = gm.summary_json()["backtest"]
    print("GAME", json.dumps({k: g[k] for k in ("n", "seasons", "model", "baselines", "total_goals")}, indent=2))
    for market, s in pm.summary_json()["backtest"].items():
        print(market.upper(), json.dumps({k: s[k] for k in ("n", "chosen_model", "calibrated", "model", "baselines", "top_decile")}))


def cmd_predict(_: argparse.Namespace) -> None:
    """Log pre-game calls (game winners and main-line props) for games that have not started."""
    from nhlpred import predict as pr
    from nhlpred.context import load
    ctx = load()
    n = pr.log_game_predictions(ctx.data.games, ctx.gpred.reset_index(), ctx.gbundle["version"], ctx.now)
    m = pr.log_prop_predictions(ctx.proj, ctx.pprob, ctx.data.games, ctx.pbundle["version"], ctx.now) if len(ctx.proj) else 0
    print(f"{n} game calls, {m} prop calls logged")


def cmd_export(_: argparse.Namespace) -> None:
    from nhlpred.export import export_all
    print(json.dumps(export_all(), indent=2, default=str))


def cmd_refresh(args: argparse.Namespace) -> None:
    """Daily job: ingest new finished games, retrain if stale, log calls, write the JSON."""
    from nhlpred import game_model as gm
    from nhlpred.pipeline import models_stale, train_all
    args.current = True
    cmd_ingest(args)
    if models_stale() or not (cfg.model_dir() / "props_model.joblib").exists() or not (cfg.model_dir() / "game_model.joblib").exists():
        log.info("models stale or missing: retraining")
        train_all()
    else:
        log.info("models fresh (%s)", gm.summary_json().get("version"))
    cmd_predict(args)
    cmd_export(args)


COMMANDS = {
    "ingest": (cmd_ingest, "pull schedule + finished boxscores (--current: running season only; default: all seasons)"),
    "train": (cmd_train, "walk-forward backtest, then fit the production game and prop models"),
    "backtest": (cmd_backtest, "run and print the walk-forward backtests (refits models too)"),
    "predict": (cmd_predict, "log pre-game calls for games that have not started"),
    "export": (cmd_export, "write the static JSON under frontend/public/data/nhl"),
    "refresh": (cmd_refresh, "daily job: ingest --current, train if stale, predict, export"),
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="nhlpred")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (fn, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--current", action="store_true", help="only the running season (ingest)")
        p.set_defaults(fn=fn)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parser.parse_args()
    args.fn(args)
