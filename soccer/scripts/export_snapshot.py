#!/usr/bin/env python3
"""Snapshot every GET the Soccer UI makes into web/snapshot/.

Runs the FastAPI app in-process (no port), reads the T7 lake, and writes each
response envelope under the filename given by `futball.snapshot_key`
(see docs/snapshot-contract.md). Run from anywhere:

    set -a; . ~/Developer/Projects/FUTBALL/.env; set +a      # provider keys, never copied
    ~/Developer/Projects/FUTBALL/.venv/bin/python scripts/export_snapshot.py [--clean]

Safe to re-run: files are overwritten in place. Quota-limited API-Football is
blocked outright; free sources (ESPN, RSS) may be hit when their cache is stale
and are counted in the summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

log = logging.getLogger("export_snapshot")

LIVE_DIR = ROOT / "web" / "snapshot"
DEFAULT_OUT = ROOT / "web" / "snapshot-new"  # built here, swapped into LIVE_DIR with --swap
BLOCKED_HOSTS = ("api-sports.io", "api-football.com")  # API-Football free tier: 100 req/day
MAX_MEMO_BYTES = 500_000  # keep parsed bodies only when small; big ones are write-only
MIN_NAME, MAX_NAME = 2, 60


# ------------------------------------------------------------------ network ---

def install_network_guard(counts: Counter) -> None:
    """Count outbound hosts and refuse the quota-limited provider."""
    real_send = httpx.Client.send

    def send(self, request, **kwargs):
        host = request.url.host
        if host != "testserver":
            counts[host] += 1
            if host.endswith(BLOCKED_HOSTS):
                raise httpx.ConnectError(f"blocked by export_snapshot: {host}")
        return real_send(self, request, **kwargs)

    httpx.Client.send = send


# ----------------------------------------------------------------- recorder ---

def _strings(node: object, sink: set[str]) -> None:
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            if MIN_NAME <= len(cur) <= MAX_NAME and not cur.startswith("http"):
                sink.add(cur)
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


@dataclass
class Recorder:
    """Fetches endpoints, writes snapshots, remembers what happened."""

    client: object
    out: Path
    snapshot_file: object
    names: set[str] = field(default_factory=set)
    sizes: dict[str, int] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    _memo: dict[str, object] = field(default_factory=dict)

    def get(self, path: str, **query: object) -> object | None:
        """GET `path`; write the snapshot on success. Returns `data` (or None)."""
        clean = {k: str(v) for k, v in query.items() if v is not None and str(v) != ""}
        rel = self.snapshot_file(path, clean)
        if rel in self._memo:
            return self._memo[rel]
        data = self._fetch_and_write(path, clean, rel)
        self._memo[rel] = data
        return data

    def _fetch_and_write(self, path: str, query: dict[str, str], rel: str) -> object | None:
        label = path + ("?" + "&".join(f"{k}={v}" for k, v in sorted(query.items())) if query else "")
        try:
            resp = self.client.get(path, params=query)
        except Exception as exc:  # noqa: BLE001 - one bad request must not stop the export
            self.skipped.append((label, f"exception {type(exc).__name__}: {exc}"[:160]))
            return None
        if resp.status_code != 200:
            self.skipped.append((label, f"HTTP {resp.status_code}"))
            return None
        if path.endswith(".ics"):
            self._write(rel, resp.content)
            return True
        try:
            body = resp.json()
        except ValueError:
            self.skipped.append((label, "not JSON"))
            return None
        if not body.get("success"):
            self.skipped.append((label, f"success=false: {str(body.get('error'))[:100]}"))
            return None
        self._write(rel, resp.content)
        _strings(body.get("data"), self.names)
        return body.get("data") if len(resp.content) <= MAX_MEMO_BYTES else True

    def write_envelope(self, path: str, data: object, query: dict[str, str] | None = None, **meta: object) -> None:
        rel = self.snapshot_file(path, query or {})
        body = {"success": True, "data": data, "error": None, "meta": meta}
        self._write(rel, json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())
        _strings(data, self.names)
        self._memo[rel] = data

    def _write(self, rel: str, content: bytes) -> None:
        dest = self.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        self.sizes[rel] = len(content)


# ---------------------------------------------------------------- phases ---

def rows(value: object) -> list:
    return value if isinstance(value, list) else []


def as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def export_globals(rec: Recorder) -> list[str]:
    """Endpoints with no dependency on a league. Returns the league codes."""
    for path in ("/health", "/teams/directory", "/predict/competitions"):
        rec.get(path)
    leagues = rows(rec.get("/leagues"))
    return [row["code"] for row in leagues]


def export_league_tables(rec: Recorder, code: str) -> tuple[list[int], dict[str, set[str]]]:
    """Standings/results for every season. Returns (season years, team names seen)."""
    years = [r["season_year"] for r in rows(rec.get(f"/leagues/{code}/seasons"))]
    teams: set[str] = set()
    for year in [None, *years]:
        standing = as_dict(rec.get(f"/leagues/{code}/standings", season_year=year))
        teams.update(r["team"] for r in rows(standing.get("rows")) if r.get("team"))
        rec.get(f"/leagues/{code}/matches", season_year=year, limit=40)
        matches = as_dict(rec.get(f"/leagues/{code}/matches", season_year=year, limit=400))
        for m in rows(matches.get("rows")):
            teams.update(t for t in (m.get("home_team"), m.get("away_team")) if t)
    return years, {code: teams}


def export_fixtures(rec: Recorder, code: str) -> list[dict]:
    """Fixture feeds, plus the head-to-head / form panels the live page opens."""
    upcoming = rows(rec.get(f"/leagues/{code}/fixtures", days=21))
    rec.get("/fixtures", competition_code=code, days=45)
    predict_list = rows(rec.get("/predict/fixture-list", competition_code=code, days=21))
    for fx in [*upcoming, *predict_list]:
        home, away = fx.get("home_team"), fx.get("away_team")
        if not (home and away):
            continue
        rec.get(f"/leagues/{code}/head-to-head", home=home, away=away, limit=6)
        for team in (home, away):
            for limit in (5, 10):
                rec.get(f"/leagues/{code}/team-form", team=team, limit=limit)
    for fx in predict_list:
        if fx.get("fixture_key"):
            rec.get("/predict/by-fixture", fixture_key=fx["fixture_key"])
    return upcoming


def export_competition_page(rec: Recorder, code: str) -> None:
    for limit in (5, 10, 20):
        rec.get("/players/leaders", competition_code=code, limit=limit)
    rec.get(f"/leagues/{code}/clean-sheets", limit=20)
    rec.get(f"/leagues/{code}/honours")
    rec.get(f"/leagues/{code}/calendar.ics")
    for limit in (14, 40):
        rec.get(f"/competitions/{code}/news", limit=limit)
    for kind, board in (("player", "player-leaders"), ("team", "team-leaders")):
        for metric in rows(rec.get(f"/competitions/{code}/{kind}-metrics")):
            rec.get(f"/competitions/{code}/{board}", metric=metric["key"], limit=3)


def export_team_of_week(rec: Recorder, code: str, years: list[int], seasons: int) -> None:
    """Default round for each of the newest `seasons`; every round for the live one."""
    for year in [None, *years[:seasons]]:
        data = as_dict(rec.get(f"/competitions/{code}/team-of-the-week", season_year=year))
        if year is None or year == years[0]:
            for rnd in rows(data.get("rounds")):
                rec.get(f"/competitions/{code}/team-of-the-week", match_week=rnd, season_year=year)


def export_predictions(rec: Recorder, code: str) -> None:
    rec.get("/predict/teams", competition_code=code)
    rec.get("/predict/metrics", competition_code=code)
    rec.get("/predict/backtest", competition_code=code, limit=12)
    board = as_dict(rec.get("/predict/matchday", competition_code=code))
    for day in rows(board.get("matchdays")):
        rec.get("/predict/matchday", competition_code=code, matchday=day)


def export_teams(rec: Recorder, team_names: dict[str, set[str]]) -> None:
    """Club pages: with the competition the UI links with, and without."""
    directory = rows(rec.get("/teams/directory"))
    free_names = {n for d in directory for n in (d.get("name"), d.get("short_name"), *d.get("aliases", [])) if n}
    seen: set[tuple[str, str | None]] = set()
    jobs = [(t, code) for code, names in team_names.items() for t in sorted(names)]
    jobs += [(t, None) for t in sorted(free_names | {t for ns in team_names.values() for t in ns})]
    for team, code in jobs:
        if (team, code) in seen:
            continue
        seen.add((team, code))
        profile = as_dict(rec.get("/teams/profile", team=team, competition=code))
        if not profile:
            continue
        if profile.get("lake_name"):
            rec.get("/teams/calendar.ics", team=profile["lake_name"])
        if profile.get("team") and profile.get("competition_code"):
            rec.get("/lineups/latest", team=profile["team"], competition_code=profile["competition_code"])


def export_players(rec: Recorder, extra_hits: list[dict]) -> int:
    """Profiles for players with a football-data record; the search index also lists box-score-only players."""
    from futball import players as players_mod
    from futball.sources import api_football

    api_football.available = lambda: False  # details(): serve cache only, never spend quota
    league_identities = [e for e in players_mod._identities().values() if e["league"]]
    known = {e["name"] for e in league_identities}
    index = [{**players_mod._search_hit(e), "norm": players_mod.normalise(e["name"])} for e in league_identities]
    index += [h for h in extra_hits if h["name"] not in known]
    rec.write_envelope("/players/search-index", index, count=len(index))
    ids = sorted(players_mod.public_id(e) for e in league_identities)
    for pid in ids:
        rec.get(f"/players/profile/{pid}")
        rec.get(f"/players/{pid}/details")
    return len(ids)


def export_props(rec: Recorder, codes: list[str]) -> tuple[dict[str, int], dict[str, dict]]:
    """props/board per competition and players/gamelog for every player on a board. Returns (counts, gamelogs)."""
    from futball import boxscores, league
    from futball.props import board as board_mod
    from futball.props import service

    stats = boxscores.load()
    if stats.height == 0:
        log.error("player_match_stats is empty: run `python -m futball.cli boxscores` first")
        return {}, {}
    stats_rows = stats.to_dicts()
    key_for = service.default_key_resolver()
    boards = service.build_boards(codes, stats_rows, league._fixtures(), service.default_predictor(), key_for)
    index = service.key_index(stats_rows, boards, key_for)
    counts: dict[str, int] = {}
    for code, board in boards.items():
        rec.write_envelope("/props/board", board, {"competition_code": code})
        counts[code] = sum(len(fx["props"]) for fx in board["fixtures"])
    # a player on two boards (domestic + UCL) is shown the fixture that comes first
    merged = {"fixtures": sorted((fx for b in boards.values() for fx in b["fixtures"]), key=lambda fx: (fx["date"], fx["time"] or ""))}
    logs: dict[str, dict] = {}
    for key, espn_id in sorted(index.items()):
        logs[key] = board_mod.gamelog(stats_rows, espn_id, key, merged)
        rec.write_envelope(f"/players/gamelog/{key}", logs[key])
    return counts, logs


# ----------------------------------------------------------------- images ---

def export_images(rec: Recorder) -> dict[str, dict[str, str]]:
    """Copy cached crests/headshots whose names appear in the exported data."""
    from futball import crests, headshots
    from futball.snapshot_key import enc

    manifest: dict[str, dict[str, str]] = {"crest": {}, "headshot": {}}
    for kind, finder in (("crest", crests.cached_path), ("headshot", headshots.cached_path)):
        canonical: dict[Path, str] = {}
        for name in sorted(rec.names):
            src = finder(name)
            if src is None:
                continue
            first = canonical.setdefault(src, name)
            rel = f"{kind}/{enc(first)}{src.suffix.lower()}"
            manifest[kind][name] = rel
            dest = rec.out / "img" / rel
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                rec.sizes[f"img/{rel}"] = src.stat().st_size
    (rec.out / "img").mkdir(parents=True, exist_ok=True)
    (rec.out / "img" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=0), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- summary ---

def summarise(out: Path, rec: Recorder, hosts: Counter, took: float) -> None:
    per_dir: Counter = Counter()
    files: Counter = Counter()
    for f in out.rglob("*"):
        if f.is_file():
            top = f.relative_to(out).parts[0] if len(f.relative_to(out).parts) > 1 else "(root)"
            per_dir[top] += f.stat().st_size
            files[top] += 1
    total = sum(per_dir.values())
    log.info("")
    log.info("snapshot: %d files, %.1f MB  (%.0fs)", sum(files.values()), total / 1e6, took)
    for top, size in per_dir.most_common():
        log.info("  %-14s %6d files  %8.1f MB", top, files[top], size / 1e6)
    log.info("outbound hosts (excluding in-process): %s", dict(hosts) or "none")
    log.info("skipped %d requests", len(rec.skipped))
    for label, why in rec.skipped:
        log.info("  SKIP %s  [%s]", label, why)


def write_meta(rec: Recorder, counts: dict[str, int]) -> None:
    meta = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": len(rec.sizes), "skipped": len(rec.skipped), **counts}
    (rec.out / "export-meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def swap_into_place(new: Path, live: Path) -> None:
    """Replace `live` with `new` by renames only, so readers see the old or the new snapshot, never a mix."""
    retired = live.with_name(live.name + "-old")
    if retired.exists():
        shutil.rmtree(retired)
    if live.exists():
        live.rename(retired)
    new.rename(live)
    log.info("swapped %s -> %s (previous kept at %s)", new.name, live.name, retired.name)


# ------------------------------------------------------------------- main ---

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clean", action="store_true", help="delete the output folder first")
    ap.add_argument("--swap", action="store_true", help="when done, replace web/snapshot/ with the output folder")
    ap.add_argument("--totw-seasons", type=int, default=1,
                    help="newest seasons whose team-of-the-week is exported (ESPN-heavy)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.time()

    hosts: Counter = Counter()
    install_network_guard(hosts)

    from fastapi.testclient import TestClient

    from futball import config
    from futball.api.app import app
    from futball.snapshot_key import snapshot_file

    try:
        config.require_volume()
    except Exception as exc:  # noqa: BLE001 - unusable data root is the one fatal case
        log.error("data root unusable: %s", exc)
        return 2
    import futball
    log.info("futball package: %s", Path(futball.__file__).parent)
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        rec = Recorder(client=client, out=args.out, snapshot_file=snapshot_file)
        codes = export_globals(rec)
        team_names: dict[str, set[str]] = {}
        for code in codes:
            log.info("league %s", code)
            years, names = export_league_tables(rec, code)
            team_names.update(names)
            export_fixtures(rec, code)
            export_competition_page(rec, code)
            export_predictions(rec, code)
            export_team_of_week(rec, code, years, args.totw_seasons)
        log.info("clubs: %d (league, team) pairs", sum(len(v) for v in team_names.values()))
        export_teams(rec, team_names)
        from futball import config as cfg
        from futball import teamcolors
        from futball.props import service
        prop_counts, logs = export_props(rec, codes)
        hits = service.espn_search_hits(logs, teamcolors.colors, {c: m["name"] for c, m in cfg.LEAGUES.items()})
        n_players = export_players(rec, hits)
        manifest = export_images(rec)

    write_meta(rec, {"players": n_players, "gamelogs": len(logs), "props": prop_counts,
                     "crests": len(set(manifest["crest"].values())),
                     "headshots": len(set(manifest["headshot"].values()))})
    summarise(args.out, rec, hosts, time.time() - started)
    log.info("images: %d crest files, %d headshot files",
             len(set(manifest["crest"].values())), len(set(manifest["headshot"].values())))
    if args.swap:
        swap_into_place(args.out, LIVE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
