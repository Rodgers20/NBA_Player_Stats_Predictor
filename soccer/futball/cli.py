"""Command line entrypoints for the data plane."""
from __future__ import annotations

import argparse
import logging
import sys

from . import config
from .sources import statsbomb


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    config.ensure_dirs()
    if args.season:
        comp_id, season_id = (int(x) for x in args.season.split(":"))
        from .http import PoliteClient

        with PoliteClient() as client:
            matches = statsbomb.fetch_matches(client, comp_id, season_id)
            selected = matches[: args.limit] if args.limit else matches
            logging.info("ingesting %d/%d matches", len(selected), len(matches))
            total = statsbomb.IngestStats()
            for i, m in enumerate(selected, 1):
                total.merge(statsbomb.fetch_match_detail(client, m["match_id"]))
                if i % 10 == 0 or i == len(selected):
                    logging.info("  %d/%d — %s", i, len(selected), total)
    else:
        total = statsbomb.ingest_360_seasons(limit_matches_per_season=args.limit)

    logging.info("DONE %s", total)
    for e in total.errors[:20]:
        logging.error("  %s", e)
    return 1 if total.failed else 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Pull the latest results, rebuild the lake and retrain.

    Safe to run daily. The 2026/27 European seasons are in progress and some
    division files had not been published upstream when this was built, so a
    re-run is how EPL / Bundesliga / Serie A appear once they go live.
    """
    from .sources import football_data as fd
    from .sources import football_data_org as fdo
    from .sources import openfootball as of
    from .transform import results
    from . import quality

    config.ensure_dirs()

    logging.info("1/5 fetching football-data.co.uk ...")
    r = fd.ingest(n_seasons=args.seasons)
    logging.info("    %s", r)
    if r.unavailable:
        logging.info("    not yet published upstream: %s", sorted(set(r.unavailable)))

    logging.info("2/5 fetching openfootball Champions League ...")
    logging.info("    %s", of.ingest_champions_league())

    # The two sources above regenerate only after a round completes, so between
    # a match finishing and that rebuild the table is a matchday behind. This
    # one closes that gap, and is the only source for Champions League 2026/27.
    logging.info("3/5 fetching football-data.org ...")
    if fdo.available():
        logging.info("    %s", fdo.ingest(ttl=0 if args.force else fdo.CACHE_TTL_SECONDS))
    else:
        logging.info("    no FOOTBALL_DATA_ORG_KEY set — skipping "
                     "(results will lag upstream by a day or two)")

    # Squads sit behind a 7-day TTL, so on most days this is a no-op; it keeps
    # portraits current through transfer windows without a manual step.
    if fdo.available():
        from . import headshots as headshot_mod
        logging.info("    caching squads for portraits ...")
        logging.info("    %s", headshot_mod.backfill_squads(limit=args.squad_limit))

    logging.info("4/5 rebuilding results layer ...")
    counts = results.build()
    for k, v in counts.items():
        logging.info("    %-14s %s", k, f"{v:,}")

    from .http import PoliteClient
    from .sources import fixtures as fixture_source
    from .transform import fixture_table
    from datetime import date
    year = date.today().year if date.today().month >= 7 else date.today().year - 1
    with PoliteClient(min_interval=0.4, max_retries=2) as client:
        fixture_source.fetch_fixtures_csv(client)
        fixture_source.fetch_season_schedules(client, [f"{year}-{str(year+1)[-2:]}"])
    logging.info("fixtures refreshed: %s", fixture_table.build())

    if args.retrain:
        logging.info("5/5 retraining prediction models ...")
        from .predict import train as train_mod
        train_mod.report()
    else:
        logging.info("5/5 skipping retrain (pass --retrain to include it)")

    # Always re-measure calibration, retrain or not: results that landed today
    # grade forecasts already served, and that is what lets the model notice it
    # has been leaning one way. It is cheap — no model is refitted.
    from .predict import calibration
    try:
        drift = calibration.refresh()
        corrected = sum(len(v) for v in drift["markets"].values())
        logging.info("calibration refreshed: %d market/competition corrections", corrected)
    except Exception:  # noqa: BLE001 - a refresh must not fail over its own bookkeeping
        logging.warning("calibration refresh failed", exc_info=True)

    return 0


def cmd_headshots(args: argparse.Namespace) -> int:
    """Cache squad rosters, then resolve portraits for known players."""
    from . import headshots, players
    from .sources import api_football as af

    config.ensure_dirs()
    remaining = af.requests_remaining()
    logging.info("api-football budget: %s request(s) left today",
                 "unknown" if remaining is None else remaining)

    logging.info("1/2 caching squads ...")
    logging.info("    %s", headshots.backfill_squads(limit=args.limit))

    if args.resolve:
        logging.info("2/2 resolving portraits ...")
        names = sorted({r["player"] for r in players._league_rows()})
        done = headshots.resolve_many(names)
        hit = sum(1 for h in done.values() if h.found)
        by_provider: dict[str, int] = {}
        for h in done.values():
            if h.found:
                by_provider[h.provider or "unknown"] = by_provider.get(h.provider or "unknown", 0) + 1
        logging.info("    %d/%d resolved %s", hit, len(names), by_provider)
    else:
        logging.info("2/2 skipping portrait download (pass --resolve to include it)")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    from . import quality
    return quality.report()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="futball")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="ingest StatsBomb open data")
    ing.add_argument("--season", help="competition_id:season_id, e.g. 55:282")
    ing.add_argument("--limit", type=int, default=None, help="max matches")
    ing.set_defaults(func=cmd_ingest)

    ref = sub.add_parser("refresh", help="update results + rebuild + retrain")
    ref.add_argument("--seasons", type=int, default=16, help="how many seasons of history")
    ref.add_argument("--retrain", action="store_true", help="retrain prediction models")
    ref.add_argument("--squad-limit", type=int, default=12,
                     help="max club squads to refresh per run (protects the "
                          "100/day API-Football budget)")
    ref.add_argument("--force", action="store_true",
                     help="ignore the football-data.org cache TTL and refetch now")
    ref.set_defaults(func=cmd_refresh)

    hs = sub.add_parser("headshots", help="cache squads and resolve player portraits")
    hs.add_argument("--limit", type=int, default=None,
                    help="max clubs to fetch this run (the rest resume next time)")
    hs.add_argument("--resolve", action="store_true",
                    help="also download portraits for every known scorer")
    hs.set_defaults(func=cmd_headshots)

    q = sub.add_parser("quality", help="run the data quality suite")
    q.set_defaults(func=cmd_quality)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except config.VolumeNotMounted as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
