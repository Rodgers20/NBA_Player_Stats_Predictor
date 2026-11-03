"""StatsBomb open-data ingestion.

This is the primary source. Verified live on 2026-08-21: 80 competition-seasons,
12 of which carry 360 freeze-frame data. A single Euro 2024 match yields ~3,610
events with full pitch coordinates, xG on every shot, and ~2,974 360-frames of
~20 positioned players — which is what lets the 3D scenes render real geometry
rather than decoration.

Layout written under raw/statsbomb/:
    competitions.json
    matches/{competition_id}/{season_id}.json
    events/{match_id}.json
    lineups/{match_id}.json
    three-sixty/{match_id}.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..storage import data_files
from ..http import PoliteClient

log = logging.getLogger(__name__)

RAW_SB = config.RAW / "statsbomb"


@dataclass
class IngestStats:
    fetched: int = 0
    cached: int = 0
    failed: int = 0
    bytes_written: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: "IngestStats") -> None:
        self.fetched += other.fetched
        self.cached += other.cached
        self.failed += other.failed
        self.bytes_written += other.bytes_written
        self.errors.extend(other.errors)

    def __str__(self) -> str:
        mb = self.bytes_written / 1_048_576
        return (
            f"fetched={self.fetched} cached={self.cached} "
            f"failed={self.failed} written={mb:.1f}MB"
        )


def _url(*parts: str) -> str:
    return "/".join((config.STATSBOMB_BASE, *parts))


def fetch_competitions(client: PoliteClient) -> list[dict]:
    dest = RAW_SB / "competitions.json"
    client.fetch_to(_url("competitions.json"), dest, skip_if_exists=False)
    return json.loads(dest.read_text())


def fetch_matches(
    client: PoliteClient, competition_id: int, season_id: int
) -> list[dict]:
    dest = RAW_SB / "matches" / str(competition_id) / f"{season_id}.json"
    client.fetch_to(
        _url("matches", str(competition_id), f"{season_id}.json"),
        dest,
        skip_if_exists=False,
    )
    return json.loads(dest.read_text())


def fetch_match_detail(
    client: PoliteClient, match_id: int, *, with_360: bool = True
) -> IngestStats:
    """Fetch events, lineups and (optionally) 360 frames for one match."""
    stats = IngestStats()
    targets = [
        ("events", RAW_SB / "events" / f"{match_id}.json", True),
        ("lineups", RAW_SB / "lineups" / f"{match_id}.json", True),
    ]
    if with_360:
        targets.append(("three-sixty", RAW_SB / "three-sixty" / f"{match_id}.json", False))

    for kind, dest, required in targets:
        try:
            res = client.fetch_to(_url(kind, f"{match_id}.json"), dest)
            if res.from_cache:
                stats.cached += 1
            else:
                stats.fetched += 1
                stats.bytes_written += res.bytes_written
        except FileNotFoundError:
            # 360 data only exists for a subset of matches; that is expected and
            # is not an error. A missing events file genuinely is.
            if required:
                stats.failed += 1
                stats.errors.append(f"missing required {kind} for match {match_id}")
        except Exception as exc:  # noqa: BLE001
            stats.failed += 1
            stats.errors.append(f"{kind} match {match_id}: {exc}")
    return stats


def ingest_360_seasons(limit_matches_per_season: int | None = None) -> IngestStats:
    """Ingest every competition-season that carries 360 data.

    `limit_matches_per_season` bounds the pull for a fast first run; None pulls
    everything. Fully resumable — already-downloaded files are skipped.
    """
    config.ensure_dirs()
    total = IngestStats()

    with PoliteClient() as client:
        comps = fetch_competitions(client)
        log.info("competitions.json: %d season entries", len(comps))
        total.fetched += 1

        for comp_id, season_id, label in config.SB_360_SEASONS:
            try:
                matches = fetch_matches(client, comp_id, season_id)
            except Exception as exc:  # noqa: BLE001
                total.failed += 1
                total.errors.append(f"matches {label}: {exc}")
                continue

            total.fetched += 1
            selected = matches[:limit_matches_per_season] if limit_matches_per_season else matches
            log.info("%s — %d matches (ingesting %d)", label, len(matches), len(selected))

            for m in selected:
                total.merge(fetch_match_detail(client, m["match_id"]))

    return total


def local_match_ids() -> list[int]:
    """Match ids whose event files are present on disk."""
    d = RAW_SB / "events"
    if not d.is_dir():
        return []
    return sorted(int(p.stem) for p in data_files(d, "*.json"))


def read_events(match_id: int) -> list[dict]:
    return json.loads((RAW_SB / "events" / f"{match_id}.json").read_text())


def read_360(match_id: int) -> list[dict]:
    p = RAW_SB / "three-sixty" / f"{match_id}.json"
    return json.loads(p.read_text()) if p.exists() else []


def read_lineups(match_id: int) -> list[dict]:
    return json.loads((RAW_SB / "lineups" / f"{match_id}.json").read_text())


def read_matches(competition_id: int, season_id: int) -> list[dict]:
    p = RAW_SB / "matches" / str(competition_id) / f"{season_id}.json"
    return json.loads(p.read_text()) if p.exists() else []


def all_local_matches() -> list[dict]:
    """Every match record we have locally, across all ingested seasons."""
    out: list[dict] = []
    d = RAW_SB / "matches"
    if not d.is_dir():
        return out
    for season_file in data_files(d, "*/*.json"):
        out.extend(json.loads(season_file.read_text()))
    return out
