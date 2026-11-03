"""Paths and constants for the NHL pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIRST_SEASON = 2024            # season = year it ends in; 2024 is 2023-24
SEASON_OPEN = (10, 1)          # month, day to start scanning the schedule
SEASON_CLOSE = (6, 30)
EXPORT_DAYS_BACK = 7
EXPORT_DAYS_AHEAD = 7
PROPS_DAYS_AHEAD = 2           # props board covers today + this many following game days
RETRAIN_DAYS = 7
FINAL_STATES = ("OFF", "FINAL")
LIVE_STATES = ("LIVE", "CRIT")
API = "https://api-web.nhle.com/v1"
MUG = "https://assets.nhle.com/mugs/nhl/{season8}/{team}/{pid}.png"


def data_dir() -> Path:
    return ROOT / "data" / "nhl"


def model_dir() -> Path:
    return ROOT / "models" / "nhl"


def export_dir() -> Path:
    return ROOT / "frontend" / "public" / "data" / "nhl"


def games_path() -> Path:
    return data_dir() / "games.parquet"


def skaters_path() -> Path:
    return data_dir() / "skater_games.parquet"


def goalies_path() -> Path:
    return data_dir() / "goalie_games.parquet"


def teams_path() -> Path:
    return data_dir() / "teams.json"


def standings_path() -> Path:
    return data_dir() / "standings.json"


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year + 1 if today.month >= 8 else today.year


def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[2:]}"
