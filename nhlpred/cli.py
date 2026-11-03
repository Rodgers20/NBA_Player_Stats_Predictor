"""Command line entry point: python -m nhlpred <command>."""

from __future__ import annotations

import argparse
import logging

from nhlpred import config as cfg


def cmd_ingest(args: argparse.Namespace) -> None:
    from nhlpred.ingest import ingest, refresh_current
    if args.current:
        print(refresh_current())
    else:
        print(ingest(list(range(cfg.FIRST_SEASON, cfg.current_season() + 1))))


COMMANDS = {
    "ingest": (cmd_ingest, "pull schedule + finished boxscores (--current: running season only)"),
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="nhlpred")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (fn, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--current", action="store_true", help="only the running season")
        p.set_defaults(fn=fn)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parser.parse_args()
    args.fn(args)
