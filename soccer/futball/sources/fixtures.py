"""Upcoming-fixture ingestion.

The results files carry only matches that have been PLAYED, so on their own the
dashboard cannot answer "what is on today", "what is live right now", or "let me
predict this weekend's game". Two sources fill that gap:

* **football-data.co.uk `fixtures.csv`** — a rolling ~4 day window across ~20
  divisions with kick-off times, referee and pre-match odds. Verified on
  2026-08-23: 165 fixtures spanning 21-24 Aug, including Newcastle v Liverpool
  at 16:30. Because it is a rolling file it is snapshotted per fetch date, so a
  fixture list is never silently rewritten.

* **openfootball season schedules** — the complete season for leagues that
  publish one (English Premier League 2026/27 has all 380 fixtures). This is what
  lets the fixture picker show games beyond the next four days.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, datetime
from typing import Any

from .. import config
from ..storage import data_files
from ..http import PoliteClient

log = logging.getLogger(__name__)

RAW_FIX = config.RAW / "fixtures"


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def fetch_fixtures_csv(client: PoliteClient) -> int:
    """Snapshot today's rolling fixture file. Returns rows written."""
    data = client.get_bytes(config.FIXTURES_URL)
    text = _decode(data)
    dest = RAW_FIX / "football-data" / f"{date.today().isoformat()}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return max(len(text.splitlines()) - 1, 0)


def fetch_season_schedules(client: PoliteClient, seasons: list[str]) -> dict[str, int]:
    """Full-season schedules from openfootball, where published."""
    out: dict[str, int] = {}
    for season in seasons:
        for div, stem in config.OF_SEASON_FILES.items():
            url = config.OPENFOOTBALL_SEASON.format(season=season, code=stem)
            dest = RAW_FIX / "openfootball" / season / f"{div}.json"
            try:
                client.fetch_to(url, dest, skip_if_exists=False)
            except Exception:  # noqa: BLE001
                continue  # not published for this league/season
            try:
                payload = json.loads(dest.read_text())
                out[f"{season}/{div}"] = len(payload.get("matches", []))
            except Exception:  # noqa: BLE001
                continue
    return out


def _parse_date(v: str | None) -> str | None:
    if not v:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(v.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _f(v: Any) -> float | None:
    if v in (None, "", "NA"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_football_data_fixtures() -> list[dict]:
    """All snapshotted rolling fixtures, most recent snapshot winning."""
    d = RAW_FIX / "football-data"
    if not d.is_dir():
        return []
    seen: dict[tuple, dict] = {}
    for path in data_files(d, "*.csv"):  # oldest first so newer overwrite
        for r in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))):
            div = (r.get("Div") or "").strip()
            iso = _parse_date(r.get("Date"))
            home = (r.get("HomeTeam") or "").strip()
            away = (r.get("AwayTeam") or "").strip()
            if not (div and iso and home and away):
                continue
            seen[(div, iso, home, away)] = {
                "competition_code": div,
                "date": iso,
                "time": (r.get("Time") or "").strip() or None,
                "home_team": home,
                "away_team": away,
                "referee": (r.get("Referee") or "").strip() or None,
                "odds_home": _f(r.get("AvgH") or r.get("B365H")),
                "odds_draw": _f(r.get("AvgD") or r.get("B365D")),
                "odds_away": _f(r.get("AvgA") or r.get("B365A")),
                "source": "football-data",
            }
    return list(seen.values())


def read_season_schedules() -> list[dict]:
    """Full-season fixtures from openfootball (played ones carry a score)."""
    d = RAW_FIX / "openfootball"
    if not d.is_dir():
        return []
    out: list[dict] = []
    for path in data_files(d, "*/*.json"):
        div = path.stem
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for m in payload.get("matches", []):
            iso = _parse_date(m.get("date"))
            if not iso:
                continue
            # openfootball mixes two score shapes across seasons:
            #   {"score": {"ft": [1, 0]}}   and   {"score": [1, 0]}
            raw_score = m.get("score")
            if isinstance(raw_score, dict):
                score = raw_score.get("ft")
            elif isinstance(raw_score, list):
                score = raw_score
            else:
                score = None
            if not (isinstance(score, list) and len(score) >= 2):
                score = None
            out.append({
                "competition_code": div,
                "date": iso,
                "time": m.get("time"),
                "home_team": (m.get("team1") or "").strip(),
                "away_team": (m.get("team2") or "").strip(),
                "round": m.get("round"),
                "home_goals": score[0] if score else None,
                "away_goals": score[1] if score else None,
                "referee": None,
                "odds_home": None, "odds_draw": None, "odds_away": None,
                "source": "openfootball",
            })
    return out


def ingest(seasons: list[str] | None = None) -> dict[str, Any]:
    config.ensure_dirs()
    seasons = seasons or ["2026-27", "2025-26"]
    report: dict[str, Any] = {}
    with PoliteClient(min_interval=0.3, max_retries=2, follow_redirects=False) as client:
        try:
            report["rolling_fixtures"] = fetch_fixtures_csv(client)
        except Exception as exc:  # noqa: BLE001
            report["rolling_fixtures_error"] = str(exc)
        report["season_schedules"] = fetch_season_schedules(client, seasons)
    return report
