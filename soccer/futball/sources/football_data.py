"""football-data.co.uk ingestion — match results, stats and closing odds.

This is the source that makes prediction possible. StatsBomb open data is rich
but historical and tournament-shaped; football-data.co.uk carries *current
season* league results for the big European divisions plus a deep history, with
shots, shots on target, corners, cards and closing betting odds on every row.
Recent seasons additionally carry match xG.

Two quirks discovered while probing on 2026-08-21, both handled here:

1. A season file that does not exist yet returns **HTTP 300 Multiple Choices**,
   not 404 — Apache content negotiation offers near-miss names (asking for
   E0.csv when only E1/E2/E3 exist). Any non-200 therefore means "not published
   yet" and is skipped rather than retried to death.
2. The files are Latin-1-ish with a UTF-8 BOM and contain trailing blank lines
   and ragged short rows near the end of a live season.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from typing import Iterator

from .. import config
from ..storage import data_files
from ..http import PoliteClient

log = logging.getLogger(__name__)

RAW_FD = config.RAW / "football-data-couk"


@dataclass
class FetchReport:
    fetched: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    bytes_written: int = 0

    def __str__(self) -> str:
        return (
            f"fetched={len(self.fetched)} not-yet-published={len(self.unavailable)} "
            f"failed={len(self.failed)} bytes={self.bytes_written/1e6:.1f}MB"
        )


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def fetch_divisions(
    client: PoliteClient, seasons: list[str], divisions: list[str] | None = None
) -> FetchReport:
    """Download season/division CSVs into the raw layer."""
    divisions = divisions or list(config.TRAINING_DIVISIONS)
    report = FetchReport()

    for season in seasons:
        for div in divisions:
            url = f"{config.FOOTBALL_DATA_BASE}/mmz4281/{season}/{div}.csv"
            dest = RAW_FD / season / f"{div}.csv"
            key = f"{season}/{div}"

            # A completed season never changes; re-downloading it is waste.
            is_current = season == config.CURRENT_SEASON_CODE
            if dest.exists() and dest.stat().st_size > 0 and not is_current:
                report.fetched.append(key)
                continue

            try:
                data = client.get_bytes(url)
            except FileNotFoundError:
                # Not published upstream yet — expected early in a season.
                report.unavailable.append(key)
                continue
            except Exception as exc:  # noqa: BLE001
                report.failed.append(f"{key}: {exc}")
                continue

            text = _decode(data)
            if not text.lstrip().lower().startswith("div"):
                # Content negotiation can return an HTML chooser with a 200.
                report.unavailable.append(key)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".partial")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(dest)
            report.fetched.append(key)
            report.bytes_written += len(data)

    return report


def fetch_extra(client: PoliteClient) -> FetchReport:
    """Download the multi-season 'extra leagues' files (MLS lives here)."""
    report = FetchReport()
    for code, (url, _name, _country) in config.FD_EXTRA.items():
        dest = RAW_FD / "extra" / f"{code}.csv"
        try:
            data = client.get_bytes(url)
        except Exception as exc:  # noqa: BLE001
            report.failed.append(f"{code}: {exc}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_decode(data), encoding="utf-8")
        report.fetched.append(code)
        report.bytes_written += len(data)
    return report


def read_division(season: str, div: str) -> list[dict]:
    """Parse one season/division CSV into clean dict rows."""
    path = RAW_FD / season / f"{div}.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
    for r in reader:
        # Live-season files end with blank/ragged rows; a row without a date and
        # both goal columns is not a played match.
        if not r.get("Date") or r.get("FTHG") in (None, "") or r.get("FTAG") in (None, ""):
            continue
        rows.append(r)
    return rows


def read_extra(code: str) -> list[dict]:
    path = RAW_FD / "extra" / f"{code}.csv"
    if not path.exists():
        return []
    rows = []
    for r in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))):
        if not r.get("Date") or r.get("HG") in (None, "") or r.get("AG") in (None, ""):
            continue
        rows.append(r)
    return rows


def available_seasons(div: str) -> list[str]:
    d = RAW_FD
    if not d.is_dir():
        return []
    return sorted(
        p.parent.name for p in data_files(d, f"*/{div}.csv", non_empty=True)
    )


def ingest(n_seasons: int = 16, divisions: list[str] | None = None) -> FetchReport:
    config.ensure_dirs()
    report = FetchReport()
    with PoliteClient(min_interval=0.4, max_retries=2, follow_redirects=True) as client:
        r1 = fetch_divisions(client, config.season_codes(n_seasons), divisions)
        r2 = fetch_extra(client)
        for r in (r1, r2):
            report.fetched += r.fetched
            report.unavailable += r.unavailable
            report.failed += r.failed
            report.bytes_written += r.bytes_written
    return report
