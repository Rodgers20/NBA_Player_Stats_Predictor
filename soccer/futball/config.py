"""Code runs locally; persistent datasets and models live on the Samsung T7."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Read the project .env regardless of the current working directory.
# Explicit process environment variables retain precedence.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

DATA_ROOT = Path(os.environ.get("FUTBALL_DATA_ROOT", "/Volumes/T7/FUTBALL-data"))

RAW = DATA_ROOT / "raw"
LAKE = DATA_ROOT / "lake"
BRONZE = LAKE / "bronze"
SILVER = LAKE / "silver"
GOLD = LAKE / "gold"
MODELS = DATA_ROOT / "models"
CACHE = DATA_ROOT / "cache"
DUCKDB_PATH = DATA_ROOT / "futball.duckdb"

# StatsBomb pitch dimensions (their coordinate system, in yards).
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Verified 2026-08-21: the twelve competition-seasons that carry 360 freeze-frame
# data, which is what makes real 3D positional rendering possible.
SB_360_SEASONS: tuple[tuple[int, int, str], ...] = (
    (55, 282, "UEFA Euro 2024"),
    (43, 106, "FIFA World Cup 2022"),
    (9, 281, "1. Bundesliga 2023/2024"),
    (11, 90, "La Liga 2020/2021"),
    (7, 235, "Ligue 1 2022/2023"),
    (7, 108, "Ligue 1 2021/2022"),
    (44, 107, "Major League Soccer 2023"),
    (55, 43, "UEFA Euro 2020"),
    (53, 315, "UEFA Women's Euro 2025"),
    (53, 106, "UEFA Women's Euro 2022"),
    (72, 107, "Women's World Cup 2023"),
    (1267, 107, "African Cup of Nations 2023"),
)


class VolumeNotMounted(RuntimeError):
    """The T7 data volume is not available."""


def require_volume() -> Path:
    """Fail fast if the T7 data volume is not mounted."""
    if not DATA_ROOT.is_dir():
        raise VolumeNotMounted(
            f"Data volume {DATA_ROOT} is not mounted.\n"
            f"Plug in the Samsung T7 and check FUTBALL_DATA_ROOT in the project .env"
        )
    marker = DATA_ROOT / "raw"
    if not marker.is_dir():
        raise VolumeNotMounted(
            f"{DATA_ROOT} is mounted but has no raw/ directory — wrong volume?"
        )
    return DATA_ROOT


def ensure_dirs() -> None:
    require_volume()
    for p in (RAW, BRONZE, SILVER, GOLD, MODELS, CACHE):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Results-based competitions (football-data.co.uk + openfootball)
#
# Verified 2026-08-21. The 2026/27 European seasons had only just kicked off, so
# some division files were not yet published upstream — La Liga (SP1) was live
# with 5 matches, while E0/D1/I1 had not appeared. The ingester is therefore
# written to be re-runnable and to treat a missing current-season file as
# "not published yet" rather than an error.
# ---------------------------------------------------------------------------

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk"

# code -> (display name, country)
FD_DIVISIONS: dict[str, tuple[str, str]] = {
    "E0":  ("Premier League", "England"),
    "SP1": ("La Liga", "Spain"),
    "D1":  ("Bundesliga", "Germany"),
    "I1":  ("Serie A", "Italy"),
    "F1":  ("Ligue 1", "France"),
}

CURRENT_SEASON_CODE = "2627"

def season_codes(n_seasons: int = 16) -> list[str]:
    """Season codes newest-first, e.g. 2627, 2526, 2425 ...

    football-data.co.uk encodes a season as the two start/end year digits.
    """
    out: list[str] = []
    start = 2026
    for i in range(n_seasons):
        y = start - i
        out.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return out

# MLS and other "extra" leagues live in a single multi-season file.
# MLS was dropped once and restored for the Soccer v2 build. The file carries goals and closing
# 1X2 odds only (no shots, SOT or corners), so MLS gets goals/1X2/BTTS models and no count markets.
FD_EXTRA: dict[str, tuple[str, str, str]] = {
    "MLS": (f"{FOOTBALL_DATA_BASE}/new/USA.csv", "Major League Soccer", "USA"),
}

OPENFOOTBALL_CL = (
    "https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/cl.txt"
)


# ---------------------------------------------------------------------------
# Upcoming fixtures
#
# football-data.co.uk publishes a rolling ~4-day fixture file with kick-off
# times and pre-match odds. openfootball publishes full-season schedules for
# some leagues. Together they give both "what is on today" and "what is on in
# October", which the results files (played matches only) cannot.
# ---------------------------------------------------------------------------

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

OPENFOOTBALL_SEASON = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
)

# division code -> openfootball file stem
OF_SEASON_FILES: dict[str, str] = {
    "E0": "en.1",
    "SP1": "es.1",
    "D1": "de.1",
    "I1": "it.1",
    "F1": "fr.1",
}

# The seven competitions the dashboard is built around, in display order.
LEAGUES: dict[str, dict[str, str]] = {
    "E0":  {"name": "Premier League", "country": "England", "kind": "league"},
    "SP1": {"name": "La Liga", "country": "Spain", "kind": "league"},
    "D1":  {"name": "Bundesliga", "country": "Germany", "kind": "league"},
    "UCL": {"name": "UEFA Champions League", "country": "Europe", "kind": "cup"},
    "MLS": {"name": "Major League Soccer", "country": "USA", "kind": "league"},
}

# ESPN site-API league slugs, shared by live scores, box-score backfill and lineups.
ESPN_SLUGS: dict[str, str] = {
    "E0": "eng.1", "SP1": "esp.1", "D1": "ger.1", "UCL": "uefa.champions", "MLS": "usa.1",
}

# Competition badges. These are FotMob's *dark-theme* variants, which are drawn
# light-on-transparent and so read directly on this dashboard's panels.
# football-data.org's equivalents are dark artwork — the Premier League mark
# measures 52/255 mean brightness — and needed a white backing plate to be
# visible at all. Measured 2026-09-13: these run 136-254/255.
COMPETITION_EMBLEMS: dict[str, str] = {
    "E0": "https://images.fotmob.com/image_resources/logo/leaguelogo/dark/47.png",
    "SP1": "https://images.fotmob.com/image_resources/logo/leaguelogo/dark/87.png",
    "UCL": "https://images.fotmob.com/image_resources/logo/leaguelogo/dark/42.png",
    "D1": "https://images.fotmob.com/image_resources/logo/leaguelogo/dark/54.png",
    "MLS": "https://images.fotmob.com/image_resources/logo/leaguelogo/dark/130.png",
}

# Where the qualification lines fall in each competition's table, as
# (last position in the band, label, semantic tone). The tones map to the
# dashboard's own palette rather than each competition's brand colours, so the
# same bar means the same thing across all three tables.
#
# The Champions League league phase is 36 clubs: 1-8 go straight to the last 16,
# 9-24 enter a knockout play-off round, 25-36 are eliminated.
QUALIFICATION_BANDS: dict[str, tuple[tuple[int, str, str], ...]] = {
    "E0": (
        (4, "UEFA Champions League", "primary"),
        (5, "UEFA Europa League", "secondary"),
        (6, "UEFA Conference League qualifying", "tertiary"),
        (17, "", ""),
        (20, "Relegation to the Championship", "danger"),
    ),
    "SP1": (
        (4, "UEFA Champions League", "primary"),
        (5, "UEFA Europa League", "secondary"),
        (6, "UEFA Conference League qualifying", "tertiary"),
        (17, "", ""),
        (20, "Relegation to Segunda Division", "danger"),
    ),
    "D1": (
        (4, "UEFA Champions League", "primary"),
        (5, "UEFA Europa League", "secondary"),
        (6, "UEFA Conference League qualifying", "tertiary"),
        (15, "", ""),
        (16, "Relegation play-off", "secondary"),
        (18, "Relegation to 2. Bundesliga", "danger"),
    ),
    # MLS: the top 9 of each conference reach the playoffs. The league table is built per
    # conference (see league.standings), so these positions are conference ranks.
    "MLS": (
        (9, "MLS Cup playoffs", "primary"),
    ),
    "UCL": (
        (8, "Qualification to 1/8 finals", "primary"),
        (24, "Knockout play-off round", "secondary"),
        (36, "Eliminated", ""),
    ),
}


def qualification_band(code: str, position: int) -> tuple[str, str] | None:
    """(label, tone) for a table position, or None outside any band."""
    for last, label, tone in QUALIFICATION_BANDS.get(code, ()):
        if position <= last:
            return (label, tone) if label else None
    return None


# Which of the dashboard's competitions have StatsBomb event data, keyed to the
# competition_id in the lake. Team of the Week and any other per-player match
# metric needs this; a competition absent here cannot have one.
#
# Empty for all three today: the lake holds UEFA Euro 2024 (competition_id 55)
# and nothing else, so the Premier League, La Liga and Champions League have no
# per-player match data to rate. Adding a competition to the lake is what fills
# this in — the code needs no other change.
STATSBOMB_COMPETITION_IDS: dict[str, int] = {}


# Divisions still INGESTED for model training even though they are not shown.
# Their compact results (about 10 MB of CSV in total) measurably improve the
# three displayed competitions: Elo and form for a club that moves between
# leagues, and simply more matches for the goal models to learn from. Set this
# to tuple(LEAGUES) to train on the displayed competitions alone.
TRAINING_DIVISIONS: tuple[str, ...] = ("E0", "SP1", "D1", "I1", "F1")

# A match is treated as in progress from kick-off until this many minutes later.
MATCH_DURATION_MINUTES = 115


def current_prediction_season() -> int:
    """Season year used when querying keyed providers."""
    from datetime import date
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1
