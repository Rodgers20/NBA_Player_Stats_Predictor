"""CBB paths and constants. Seasons are named by the year they end in (2027 = the 2026-27 season),
which is what ESPN calls ``season.year``. Data lives in ``data/cbb`` and models in ``models/cbb``.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

FIRST_SEASON = 2023          # 2022-23: ratings burn-in, never used as model training or backtest rows
FIRST_BACKTEST_SEASON = 2025  # first season predicted by a model fit on earlier seasons only
D1_GROUP = "50"
SEASON_TYPES = (2, 3)         # regular season and postseason; ESPN type 1 is preseason exhibitions
SEASON_OPEN = (10, 25)        # (month, day) of the first date scanned, in the calendar year before the season ends
SEASON_CLOSE = (4, 20)        # last date scanned, in the season's end year
AVG_POSSESSIONS = 68.0        # D1 average possessions per team per game, to express points as per-100
UNRANKED = 99                 # ESPN's curatedRank for a team outside the AP Top 25
POLL_SIZE = 25

# Export window around today and how many recent graded games the performance file lists.
EXPORT_DAYS_BACK = 14
EXPORT_DAYS_AHEAD = 7
RECENT_GRADED = 15
STALE_LOOKBACK_DAYS = 30       # a game still 'scheduled' this long after its date is abandoned, not re-scanned forever
LINE_LOOKAHEAD_DAYS = 8       # upcoming days whose sportsbook lines are refreshed on each run

# Line providers in order of preference for the number we treat as "the" line; others rank after these.
PROVIDER_ORDER = ("DraftKings", "ESPN BET", "consensus", "Caesars Sportsbook", "MGM")
# ESPN's odds page also lists projection services and in-play ("Live Odds") feeds; neither is a pre-game market.
NOT_PREGAME_BOOKS = ("accuscore", "betegy", "teamrankings", "live odds")


def data_dir() -> Path:
    return Path(os.environ.get("CBB_DATA_DIR", _ROOT / "data" / "cbb"))


def model_dir() -> Path:
    return Path(os.environ.get("CBB_MODEL_DIR", _ROOT / "models" / "cbb"))


def export_dir() -> Path:
    return Path(os.environ.get("CBB_EXPORT_DIR", _ROOT / "frontend" / "public" / "data" / "cbb"))


def games_path() -> Path:
    return data_dir() / "games.parquet"


def teams_path() -> Path:
    return data_dir() / "teams.parquet"


def conferences_path() -> Path:
    return data_dir() / "conferences.json"


def team_conf_path() -> Path:
    return data_dir() / "team_conferences.json"


def ap_poll_path() -> Path:
    return data_dir() / "ap_poll.json"


def odds_history_path() -> Path:
    return data_dir() / "odds_history.parquet"


def odds_checked_path() -> Path:
    return data_dir() / "odds_checked.json"


def ratings_path() -> Path:
    return data_dir() / "ratings.parquet"


def team_ratings_path() -> Path:
    return data_dir() / "team_ratings.parquet"


def predictions_path() -> Path:
    return data_dir() / "predictions.parquet"


def backtest_path() -> Path:
    return data_dir() / "backtest.parquet"


def graded_path() -> Path:
    return data_dir() / "graded.parquet"
