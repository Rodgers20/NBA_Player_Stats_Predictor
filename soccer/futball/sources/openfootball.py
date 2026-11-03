"""openfootball ingestion — UEFA Champions League results.

football-data.co.uk covers domestic divisions only, so European club
competitions need another source. openfootball publishes the Champions League
as a structured plain-text file (public domain, no API key), which this module
parses.

Format, by inspection of the 2025/26 file:

    = UEFA Champions League 2025/26
    ▪ League, Matchday 1
      Tue Sep 16 2025
        18:45  Athletic Club (ESP)   v Arsenal FC (ENG)     0-2 (0-0)
               PSV (NED)             v Union SG (BEL)       1-3 (0-2)
      Wed Sep 17

Two things make this fiddly and are handled explicitly: the year appears only on
the first date of a block and must be carried forward, and the kick-off time is
present on the first fixture of a slot only, with subsequent fixtures indented
under it.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

from .. import config
from ..storage import data_files
from ..http import PoliteClient

log = logging.getLogger(__name__)

RAW_OF = config.RAW / "openfootball"

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# "Tue Sep 16 2025" / "Wed Sep 17"
DATE_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
STAGE_RE = re.compile(r"^\s*[▪•]\s*(.+?)\s*$")
# optional "18:45", then "Home (XXX) v Away (XXX)  2-1 (1-0)"
MATCH_RE = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)(?:\s+\((\d+)-(\d+)\))?\s*$"
)
COUNTRY_RE = re.compile(r"\s*\(([A-Z]{3})\)\s*$")


def _clean_team(raw: str) -> tuple[str, str | None]:
    m = COUNTRY_RE.search(raw)
    country = m.group(1) if m else None
    name = COUNTRY_RE.sub("", raw).strip()
    return name, country


def parse_cl(text: str, season_label: str) -> list[dict]:
    rows: list[dict] = []
    year: int | None = None
    month: int | None = None
    day: int | None = None
    stage = "League"

    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("=", "#")):
            continue

        st = STAGE_RE.match(line)
        if st and " v " not in line:
            stage = st.group(1)
            continue

        dm = DATE_RE.match(line)
        if dm:
            month = MONTHS.get(dm.group(1))
            day = int(dm.group(2))
            if dm.group(3):
                year = int(dm.group(3))
            continue

        mm = MATCH_RE.match(line)
        if mm and year and month and day:
            time_s, home_raw, away_raw, hg, ag, hth, hta = mm.groups()
            home, home_cc = _clean_team(home_raw)
            away, away_cc = _clean_team(away_raw)
            if not home or not away:
                continue
            try:
                date = datetime(year, month, day).date().isoformat()
            except ValueError:
                continue
            rows.append({
                "date": date, "time": time_s, "stage": stage,
                "home_team": home, "away_team": away,
                "home_country": home_cc, "away_country": away_cc,
                "home_goals": int(hg), "away_goals": int(ag),
                "ht_home_goals": int(hth) if hth else None,
                "ht_away_goals": int(hta) if hta else None,
                "season": season_label,
            })
    return rows


AVAILABLE_CL_SEASONS = (
    "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
)


def ingest_champions_league() -> dict[str, int]:
    """Fetch every Champions League season openfootball publishes."""
    config.ensure_dirs()
    RAW_OF.mkdir(parents=True, exist_ok=True)
    counts = {"seasons": 0, "matches": 0, "missing": 0}

    with PoliteClient(min_interval=0.3, max_retries=2) as client:
        for season in AVAILABLE_CL_SEASONS:
            url = config.OPENFOOTBALL_CL.format(season=season)
            dest = RAW_OF / "champions-league" / f"{season}.txt"
            try:
                client.fetch_to(url, dest, skip_if_exists=False)
            except Exception:  # noqa: BLE001
                # A future season simply has no file yet.
                counts["missing"] += 1
                continue
            counts["seasons"] += 1
            counts["matches"] += len(parse_cl(dest.read_text(), season))
    return counts


def read_champions_league() -> list[dict]:
    d = RAW_OF / "champions-league"
    if not d.is_dir():
        return []
    out: list[dict] = []
    for path in data_files(d, "*.txt"):
        out.extend(parse_cl(path.read_text(), path.stem))
    return out
