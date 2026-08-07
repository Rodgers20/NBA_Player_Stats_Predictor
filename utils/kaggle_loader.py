"""
Kaggle NBA Dataset Loader
=========================
Downloads and transforms the Kaggle NBA dataset to match
the existing pipeline's expected column schemas.

The Kaggle dataset lives in kagglehub's cache (outside the repo).
This module transforms Kaggle columns -> pipeline-expected columns.

Dataset: eoinamoore/historical-nba-data-and-player-box-scores
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

from utils.league_config import get_config

# Kept as module-level for backward compat with existing NBA callers.
# Step 2 refactor introduces a `league` parameter to every public function.
KAGGLE_DATASET = get_config("nba").kaggle_dataset
PROJECT_DATA_DIR = get_config("nba").data_dir

# Honor KAGGLEHUB_CACHE env var (kagglehub defaults to ~/.cache/kagglehub when unset)
KAGGLEHUB_CACHE_DIR = Path(os.environ.get("KAGGLEHUB_CACHE", Path.home() / ".cache" / "kagglehub"))

# NBA team IDs start with this prefix
_NBA_TEAM_ID_MIN = 1610612737
_NBA_TEAM_ID_MAX = 1610612766

# Cache for team abbreviation lookup (built lazily)
_team_abbrev_cache: Optional[dict] = None
_dataset_path: Optional[Path] = None


# =============================================================================
# DATASET DOWNLOAD
# =============================================================================


def download_dataset(force: bool = False) -> Path:
    """
    Download/update the Kaggle NBA dataset.
    Validates the downloaded directory is non-empty before returning.

    Returns: Path to the downloaded dataset directory.
    Raises: FileNotFoundError if download succeeds but PlayerStatistics.csv is missing.
    """
    import kagglehub

    global _dataset_path
    path = kagglehub.dataset_download(KAGGLE_DATASET, force_download=force)
    result_path = Path(path)

    # Validate the download actually contains data
    csv_file = result_path / "PlayerStatistics.csv"
    if not csv_file.exists() or csv_file.stat().st_size < 1_000_000:
        raise FileNotFoundError(
            f"Kaggle download appears corrupted or empty: {result_path}\n"
            f"PlayerStatistics.csv missing or <1MB. "
            f"Delete {KAGGLEHUB_CACHE_DIR}/datasets/eoinamoore/ and retry."
        )

    size_mb = csv_file.stat().st_size / 1e6
    logger.info(f"Dataset OK: {result_path} (PlayerStatistics.csv = {size_mb:.0f} MB)")
    _dataset_path = result_path
    return result_path


def _get_dataset_path() -> Path:
    """Get cached dataset path, downloading if needed. Works offline."""
    global _dataset_path
    if _dataset_path is not None:
        return _dataset_path

    # Try downloading from Kaggle
    try:
        _dataset_path = download_dataset()
        return _dataset_path
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")

    # Fallback: scan local kagglehub cache for latest version
    cache_base = KAGGLEHUB_CACHE_DIR / "datasets" / "eoinamoore" / "historical-nba-data-and-player-box-scores" / "versions"
    if cache_base.exists():
        versions = sorted(cache_base.iterdir(), key=lambda p: p.name, reverse=True)
        for v in versions:
            if (v / "PlayerStatistics.csv").exists():
                _dataset_path = v
                logger.info(f"Using cached dataset: {_dataset_path}")
                return _dataset_path

    # Last resort: use exported CSVs in data/ directory
    if (PROJECT_DATA_DIR / "player_game_logs.csv").exists():
        logger.info(f"Using exported data dir: {PROJECT_DATA_DIR}")
        _dataset_path = PROJECT_DATA_DIR
        return _dataset_path

    raise FileNotFoundError("No Kaggle dataset found. Run: python scripts/setup_kaggle_data.py")


# =============================================================================
# SEASON UTILITIES
# =============================================================================


def _date_to_season(dt: datetime) -> str:
    """Convert a datetime to NBA season string like '2025-26'."""
    year = dt.year if dt.month >= 10 else dt.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _date_to_season_id(dt: datetime) -> int:
    """Convert a datetime to NBA season ID like 22025."""
    year = dt.year if dt.month >= 10 else dt.year - 1
    return 20000 + year


def get_recent_seasons(num_seasons: int = 3) -> list[str]:
    """Get list of recent NBA season strings."""
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    return [f"{start_year - i}-{str(start_year - i + 1)[-2:]}"
            for i in range(num_seasons)]


# =============================================================================
# TEAM ABBREVIATION LOOKUP
# =============================================================================


def _build_team_abbrev_lookup() -> dict:
    """
    Build (teamCity, teamName) -> teamAbbrev mapping from TeamHistories.csv.
    Only includes NBA teams (not international/All-Star).
    Returns dict keyed by (city, name) tuple.
    """
    global _team_abbrev_cache
    if _team_abbrev_cache is not None and len(_team_abbrev_cache) > 0:
        return _team_abbrev_cache

    path = _get_dataset_path() / "TeamHistories.csv"
    th = pd.read_csv(path)

    # Filter to NBA teams only
    th = th[(th["teamId"] >= _NBA_TEAM_ID_MIN) & (th["teamId"] <= _NBA_TEAM_ID_MAX)]

    # Take the most recent entry per team
    th = th.sort_values("seasonActiveTill", ascending=False)
    th = th.drop_duplicates(subset=["teamId"], keep="first")

    # Build lookup: (city, name) -> abbrev and also teamId -> abbrev
    lookup = {}
    for _, row in th.iterrows():
        abbrev = str(row["teamAbbrev"]).strip()
        lookup[(row["teamCity"], row["teamName"])] = abbrev
        lookup[row["teamId"]] = abbrev

    # Also add historical entries for teams that changed cities/names
    # so we can map older game data correctly
    th_all = pd.read_csv(path)
    th_all = th_all[(th_all["teamId"] >= _NBA_TEAM_ID_MIN) & (th_all["teamId"] <= _NBA_TEAM_ID_MAX)]
    for _, row in th_all.iterrows():
        key = (row["teamCity"], row["teamName"])
        if key not in lookup:
            lookup[key] = str(row["teamAbbrev"]).strip()

    _team_abbrev_cache = lookup
    return lookup


def _get_team_abbrev(city: str, name: str) -> str:
    """Get team abbreviation from city and name."""
    lookup = _build_team_abbrev_lookup()
    abbrev = lookup.get((city, name))
    if abbrev is None:
        # Fallback: try matching just the name
        for key, a in lookup.items():
            if isinstance(key, tuple) and key[1] == name:
                return a
        logger.warning(f"No abbreviation for {city} {name}")
        return "UNK"
    return abbrev


def _get_team_abbrev_by_id(team_id: int) -> str:
    """Get team abbreviation from team ID."""
    lookup = _build_team_abbrev_lookup()
    abbrev = lookup.get(team_id)
    if abbrev is None:
        logger.warning(f"No abbreviation for team_id {team_id}")
        return "UNK"
    return abbrev


# =============================================================================
# SCHEMA ENFORCEMENT
# =============================================================================


EXPECTED_PLAYER_LOG_COLUMNS = {
    "SEASON_ID": 0, "Player_ID": 0, "Game_ID": "",
    "GAME_DATE": "", "MATCHUP": "", "WL": "",
    "MIN": 0.0, "FGM": 0, "FGA": 0, "FG_PCT": 0.0,
    "FG3M": 0, "FG3A": 0, "FG3_PCT": 0.0,
    "FTM": 0, "FTA": 0, "FT_PCT": 0.0,
    "OREB": 0, "DREB": 0, "REB": 0, "AST": 0,
    "STL": 0, "BLK": 0, "TOV": 0, "PF": 0,
    "PTS": 0, "PLUS_MINUS": 0.0, "VIDEO_AVAILABLE": 0,
    "PLAYER_NAME": "", "SEASON": "",
}


def _ensure_schema(df: pd.DataFrame, expected: dict) -> pd.DataFrame:
    """Ensure all expected columns exist; add missing with defaults."""
    for col, default in expected.items():
        if col not in df.columns:
            logger.warning(f"Missing column '{col}', filling with default: {default}")
            df[col] = default
    return df


# =============================================================================
# PLAYER GAME LOGS
# =============================================================================


def load_player_game_logs(num_seasons: int = 3) -> pd.DataFrame:
    """
    Load player game-by-game box scores from Kaggle dataset.

    Returns DataFrame matching the existing player_game_logs.csv schema:
    SEASON_ID, Player_ID, Game_ID, GAME_DATE, MATCHUP, WL, MIN,
    FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT,
    OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, PLUS_MINUS,
    VIDEO_AVAILABLE, PLAYER_NAME, SEASON
    """
    csv_path = _get_dataset_path() / "PlayerStatistics.csv"
    logger.info(f"Loading player stats from {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Include Regular Season and Playoffs (so playoff game logs appear in charts)
    df = df[df["gameType"].isin(["Regular Season", "Playoffs", "Play-in Tournament"])]

    # Parse dates and filter to recent seasons
    df["_dt"] = pd.to_datetime(df["gameDateTimeEst"])
    recent = get_recent_seasons(num_seasons)
    df["SEASON"] = df["_dt"].apply(_date_to_season)
    df = df[df["SEASON"].isin(recent)]

    logger.info(f"Filtered to {len(df)} rows across seasons: {recent}")

    # Build team abbreviation lookups for matchup strings (vectorized)
    lookup = _build_team_abbrev_lookup()
    team_pairs = df[["playerteamCity", "playerteamName"]].drop_duplicates()
    opp_pairs = df[["opponentteamCity", "opponentteamName"]].drop_duplicates()

    team_map = {(r["playerteamCity"], r["playerteamName"]):
                _get_team_abbrev(r["playerteamCity"], r["playerteamName"])
                for _, r in team_pairs.iterrows()}
    opp_map = {(r["opponentteamCity"], r["opponentteamName"]):
               _get_team_abbrev(r["opponentteamCity"], r["opponentteamName"])
               for _, r in opp_pairs.iterrows()}

    df["_player_team_abbrev"] = df.apply(
        lambda r: team_map.get((r["playerteamCity"], r["playerteamName"]), "UNK"),
        axis=1
    )
    df["_opp_team_abbrev"] = df.apply(
        lambda r: opp_map.get((r["opponentteamCity"], r["opponentteamName"]), "UNK"),
        axis=1
    )

    # Build MATCHUP string: "TOR vs. CHI" (home) or "TOR @ CHI" (away)
    df["MATCHUP"] = df.apply(
        lambda r: (f"{r['_player_team_abbrev']} vs. {r['_opp_team_abbrev']}"
                   if r["home"] == 1
                   else f"{r['_player_team_abbrev']} @ {r['_opp_team_abbrev']}"),
        axis=1
    )

    # Rename columns to match pipeline schema
    rename_map = {
        "personId": "Player_ID",
        "gameId": "Game_ID",
        "numMinutes": "MIN",
        "points": "PTS",
        "assists": "AST",
        "blocks": "BLK",
        "steals": "STL",
        "fieldGoalsAttempted": "FGA",
        "fieldGoalsMade": "FGM",
        "fieldGoalsPercentage": "FG_PCT",
        "threePointersAttempted": "FG3A",
        "threePointersMade": "FG3M",
        "threePointersPercentage": "FG3_PCT",
        "freeThrowsAttempted": "FTA",
        "freeThrowsMade": "FTM",
        "freeThrowsPercentage": "FT_PCT",
        "reboundsDefensive": "DREB",
        "reboundsOffensive": "OREB",
        "reboundsTotal": "REB",
        "foulsPersonal": "PF",
        "turnovers": "TOV",
        "plusMinusPoints": "PLUS_MINUS",
    }
    df = df.rename(columns=rename_map)

    # Build derived columns
    df["PLAYER_NAME"] = df["firstName"] + " " + df["lastName"]
    df["GAME_DATE"] = df["_dt"].dt.strftime("%b %d, %Y")
    df["WL"] = df["win"].map({1: "W", 0: "L"})
    df["SEASON_ID"] = df["_dt"].apply(_date_to_season_id)
    df["VIDEO_AVAILABLE"] = 0

    # Select and order columns to match existing schema
    output_cols = list(EXPECTED_PLAYER_LOG_COLUMNS.keys())
    df = _ensure_schema(df, EXPECTED_PLAYER_LOG_COLUMNS)
    df = df[output_cols]

    return df


# =============================================================================
# TEAM STATS (OFFENSIVE / PACE)
# =============================================================================


def load_team_stats(num_seasons: int = 3) -> pd.DataFrame:
    """
    Load and aggregate team stats from Kaggle TeamStatistics.csv.

    Returns DataFrame matching the existing team_stats.csv schema:
    TEAM_ID, TEAM_NAME, GP, W, L, W_PCT, MIN, FGM, FGA, FG_PCT, ...
    PLUS_MINUS, + rank columns, SEASON
    """
    csv_path = _get_dataset_path() / "TeamStatistics.csv"
    logger.info(f"Loading team stats from {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse dates and filter to recent seasons
    df["_dt"] = pd.to_datetime(df["gameDateTimeEst"])
    recent = get_recent_seasons(num_seasons)
    df["SEASON"] = df["_dt"].apply(_date_to_season)
    df = df[df["SEASON"].isin(recent)]

    # Filter to NBA teams only
    df = df[(df["teamId"] >= _NBA_TEAM_ID_MIN) & (df["teamId"] <= _NBA_TEAM_ID_MAX)]

    # Filter to Regular Season using Games.csv gameType
    games = pd.read_csv(_get_dataset_path() / "Games.csv",
                        usecols=["gameId", "gameType"])
    regular_ids = set(games[games["gameType"] == "Regular Season"]["gameId"])
    df = df[df["gameId"].isin(regular_ids)]

    # Aggregate per team per season
    agg = df.groupby(["teamId", "teamCity", "teamName", "SEASON"]).agg(
        GP=("gameId", "count"),
        W=("win", "sum"),
        MIN=("numMinutes", "mean"),
        FGM=("fieldGoalsMade", "mean"),
        FGA=("fieldGoalsAttempted", "mean"),
        FG3M=("threePointersMade", "mean"),
        FG3A=("threePointersAttempted", "mean"),
        FTM=("freeThrowsMade", "mean"),
        FTA=("freeThrowsAttempted", "mean"),
        OREB=("reboundsOffensive", "mean"),
        DREB=("reboundsDefensive", "mean"),
        REB=("reboundsTotal", "mean"),
        AST=("assists", "mean"),
        TOV=("turnovers", "mean"),
        STL=("steals", "mean"),
        BLK=("blocks", "mean"),
        PF=("foulsPersonal", "mean"),
        PTS=("teamScore", "mean"),
        PLUS_MINUS=("plusMinusPoints", "mean"),
    ).reset_index()

    # Compute derived columns
    agg["L"] = agg["GP"] - agg["W"]
    agg["W_PCT"] = (agg["W"] / agg["GP"]).round(3)
    agg["FG_PCT"] = (agg["FGM"] / agg["FGA"]).round(3)
    agg["FG3_PCT"] = (agg["FG3M"] / agg["FG3A"]).round(3)
    agg["FT_PCT"] = (agg["FTM"] / agg["FTA"]).round(3)

    # Estimate PACE: 48 * (FGA + 0.44*FTA - OREB + TOV) / (MIN/5)
    agg["PACE"] = (48 * (agg["FGA"] + 0.44 * agg["FTA"] - agg["OREB"] + agg["TOV"])
                   / (agg["MIN"] / 5)).round(1)

    # Round per-game averages
    avg_cols = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF",
                "PTS", "PLUS_MINUS"]
    for col in avg_cols:
        agg[col] = agg[col].round(1)

    # Add rank columns (rank within each season)
    rank_cols = ["GP", "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT",
                 "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
                 "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK",
                 "PF", "PTS", "PLUS_MINUS"]
    # Higher is better for most stats (ascending=False)
    ascending_map = {"L": True, "TOV": True, "PF": True}
    for col in rank_cols:
        asc = ascending_map.get(col, False)
        agg[f"{col}_RANK"] = agg.groupby("SEASON")[col].rank(
            ascending=asc, method="min"
        ).astype(int)

    # Also add BLKA, PFD columns with defaults (not in Kaggle data)
    agg["BLKA"] = 0.0
    agg["PFD"] = 0.0
    agg["BLKA_RANK"] = 1
    agg["PFD_RANK"] = 1

    # Add team abbreviation (useful for pace feature joins)
    agg["TEAM_ABBREVIATION"] = agg["teamId"].apply(_get_team_abbrev_by_id)

    # Rename to match schema
    agg = agg.rename(columns={
        "teamId": "TEAM_ID",
        "teamName": "TEAM_NAME",
    })

    # Build TEAM_NAME as "City Name" format
    agg["TEAM_NAME"] = agg["teamCity"] + " " + agg["TEAM_NAME"]
    agg = agg.drop(columns=["teamCity"])

    return agg


# =============================================================================
# TEAM DEFENSIVE STATS
# =============================================================================


def load_team_defensive_stats(num_seasons: int = 3) -> pd.DataFrame:
    """
    Compute opponent stats from Kaggle data.

    For each team, finds what opponents scored/shot against them,
    matching the existing team_defensive_stats.csv schema.
    """
    csv_path = _get_dataset_path() / "TeamStatistics.csv"
    logger.info(f"Loading team defensive stats from {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse dates and filter
    df["_dt"] = pd.to_datetime(df["gameDateTimeEst"])
    recent = get_recent_seasons(num_seasons)
    df["SEASON"] = df["_dt"].apply(_date_to_season)
    df = df[df["SEASON"].isin(recent)]

    # Filter to NBA teams
    df = df[(df["teamId"] >= _NBA_TEAM_ID_MIN) & (df["teamId"] <= _NBA_TEAM_ID_MAX)]

    # Filter to Regular Season
    games = pd.read_csv(_get_dataset_path() / "Games.csv",
                        usecols=["gameId", "gameType"])
    regular_ids = set(games[games["gameType"] == "Regular Season"]["gameId"])
    df = df[df["gameId"].isin(regular_ids)]

    # For defensive stats: group by opponentTeamId
    # When opponentTeamId = X, that row shows what some team scored AGAINST X
    # So averaging by opponentTeamId gives X's defensive stats
    agg = df.groupby(["opponentTeamId", "SEASON"]).agg(
        GP=("gameId", "count"),
        W=("win", lambda x: (x == 0).sum()),  # Opponent wins = team losses
        OPP_FGM=("fieldGoalsMade", "mean"),
        OPP_FGA=("fieldGoalsAttempted", "mean"),
        OPP_FG3M=("threePointersMade", "mean"),
        OPP_FG3A=("threePointersAttempted", "mean"),
        OPP_FTM=("freeThrowsMade", "mean"),
        OPP_FTA=("freeThrowsAttempted", "mean"),
        OPP_OREB=("reboundsOffensive", "mean"),
        OPP_DREB=("reboundsDefensive", "mean"),
        OPP_REB=("reboundsTotal", "mean"),
        OPP_AST=("assists", "mean"),
        OPP_TOV=("turnovers", "mean"),
        OPP_STL=("steals", "mean"),
        OPP_BLK=("blocks", "mean"),
        OPP_PF=("foulsPersonal", "mean"),
        OPP_PTS=("teamScore", "mean"),
        PLUS_MINUS=("plusMinusPoints", "mean"),
        MIN=("numMinutes", "mean"),
    ).reset_index()

    # Compute derived
    agg["L"] = agg["GP"] - agg["W"]
    agg["W_PCT"] = (agg["W"] / agg["GP"]).round(3)
    agg["OPP_FG_PCT"] = (agg["OPP_FGM"] / agg["OPP_FGA"]).round(3)
    agg["OPP_FG3_PCT"] = (agg["OPP_FG3M"] / agg["OPP_FG3A"]).round(3)
    agg["OPP_FT_PCT"] = (agg["OPP_FTM"] / agg["OPP_FTA"]).round(3)
    agg["PLUS_MINUS"] = (-agg["PLUS_MINUS"]).round(1)  # Flip sign for defensive perspective

    # Round averages (keep 3 decimals for percentages)
    pct_cols = {"OPP_FG_PCT", "OPP_FG3_PCT", "OPP_FT_PCT"}
    opp_avg_cols = [c for c in agg.columns if c.startswith("OPP_") or c == "MIN"]
    for col in opp_avg_cols:
        if col in pct_cols:
            continue  # Already rounded to 3 decimals above
        agg[col] = agg[col].round(1)

    # Add rank columns
    rank_cols = ["GP", "W", "L", "W_PCT", "MIN",
                 "OPP_FGM", "OPP_FGA", "OPP_FG_PCT",
                 "OPP_FG3M", "OPP_FG3A", "OPP_FG3_PCT",
                 "OPP_FTM", "OPP_FTA", "OPP_FT_PCT",
                 "OPP_OREB", "OPP_DREB", "OPP_REB",
                 "OPP_AST", "OPP_TOV", "OPP_STL", "OPP_BLK",
                 "OPP_PF", "OPP_PTS", "PLUS_MINUS"]
    # For defensive stats, LOWER opponent stats = better defense
    # So ascending=True for opponent stats (rank 1 = lowest = best defense)
    for col in rank_cols:
        asc = True if col.startswith("OPP_") else False
        agg[f"{col}_RANK"] = agg.groupby("SEASON")[col].rank(
            ascending=asc, method="min"
        ).astype(int)

    # BLKA and PFD not available
    agg["OPP_BLKA"] = 0.0
    agg["OPP_PFD"] = 0.0
    agg["OPP_BLKA_RANK"] = 1
    agg["OPP_PFD_RANK"] = 1

    # Get team identifiers from TeamHistories
    agg["TEAM_ID"] = agg["opponentTeamId"]
    agg["TEAM_NAME"] = agg["TEAM_ID"].apply(_get_team_name)
    agg["TEAM_ABBREVIATION"] = agg["TEAM_ID"].apply(_get_team_abbrev_by_id)
    agg = agg.drop(columns=["opponentTeamId"])

    return agg


def _get_team_name(team_id: int) -> str:
    """Get team full name from team ID via TeamHistories."""
    path = _get_dataset_path() / "TeamHistories.csv"
    th = pd.read_csv(path)
    row = th[(th["teamId"] == team_id)].sort_values("seasonActiveTill", ascending=False)
    if len(row) > 0:
        r = row.iloc[0]
        return f"{r['teamCity']} {r['teamName']}"
    return f"Team {team_id}"


# =============================================================================
# PLAYER POSITIONS
# =============================================================================


def load_player_positions(num_seasons: int = 3) -> pd.DataFrame:
    """
    Load player positions from Kaggle data.

    Returns DataFrame matching player_positions.csv schema:
    PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, POSITION, SEASON
    """
    # Load player stats to infer positions from stats (same as existing pipeline)
    # and get current team assignments
    stats_path = _get_dataset_path() / "PlayerStatistics.csv"
    stats = pd.read_csv(stats_path,
                        usecols=["personId", "firstName", "lastName",
                                 "gameDateTimeEst", "playerteamCity",
                                 "playerteamName", "gameType",
                                 "assists", "reboundsTotal"],
                        low_memory=False)
    stats = stats[stats["gameType"] == "Regular Season"].copy()
    stats["_dt"] = pd.to_datetime(stats["gameDateTimeEst"])

    # Filter to recent seasons
    recent = get_recent_seasons(num_seasons)
    stats["SEASON"] = stats["_dt"].apply(_date_to_season)
    stats = stats[stats["SEASON"].isin(recent)]

    # Compute per-game averages for position inference
    player_avgs = stats.groupby(["personId", "SEASON"]).agg(
        AST_AVG=("assists", "mean"),
        REB_AVG=("reboundsTotal", "mean"),
    ).reset_index()

    # Infer position using same heuristic as existing pipeline
    def _infer_position(row):
        ast = row["AST_AVG"]
        reb = row["REB_AVG"]
        if ast >= 4.0 and reb < 5.0:
            return "G"
        elif reb >= 7.0:
            return "C"
        return "F"

    player_avgs["POSITION"] = player_avgs.apply(_infer_position, axis=1)

    # Get most recent team per player per season
    stats = stats.sort_values("_dt", ascending=False)
    latest = stats.drop_duplicates(subset=["personId", "SEASON"], keep="first").copy()
    latest["TEAM_ABBREVIATION"] = latest.apply(
        lambda r: _get_team_abbrev(r["playerteamCity"], r["playerteamName"]),
        axis=1
    )
    latest["PLAYER_NAME"] = latest["firstName"] + " " + latest["lastName"]

    # Merge position
    result = latest[["personId", "PLAYER_NAME", "TEAM_ABBREVIATION", "SEASON"]].merge(
        player_avgs[["personId", "SEASON", "POSITION"]],
        on=["personId", "SEASON"],
        how="left"
    )
    result["POSITION"] = result["POSITION"].fillna("F")

    result = result.rename(columns={"personId": "PLAYER_ID"})
    result["PLAYER_ID"] = result["PLAYER_ID"].astype(int)
    result = result[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "POSITION", "SEASON"]]
    result = result.dropna(subset=["PLAYER_NAME"])

    return result


# =============================================================================
# CONVENIENCE: EXPORT TO PROJECT DATA DIR
# =============================================================================


def export_pipeline_csvs(num_seasons: int = 3) -> None:
    """
    Load all Kaggle data and export as pipeline-ready CSVs to data/.
    This is the main entry point for refreshing training data.
    """
    from utils.data_fetch import calculate_defense_vs_position

    PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading player game logs...")
    game_logs = load_player_game_logs(num_seasons)
    game_logs.to_csv(PROJECT_DATA_DIR / "player_game_logs.csv", index=False)
    print(f"  -> {len(game_logs)} rows, {game_logs['PLAYER_NAME'].nunique()} players")

    print("Loading team stats...")
    team_stats = load_team_stats(num_seasons)
    team_stats.to_csv(PROJECT_DATA_DIR / "team_stats.csv", index=False)
    print(f"  -> {len(team_stats)} rows")

    print("Loading team defensive stats...")
    team_def = load_team_defensive_stats(num_seasons)
    team_def.to_csv(PROJECT_DATA_DIR / "team_defensive_stats.csv", index=False)
    print(f"  -> {len(team_def)} rows")

    print("Loading player positions...")
    positions = load_player_positions(num_seasons)
    positions.to_csv(PROJECT_DATA_DIR / "player_positions.csv", index=False)
    print(f"  -> {len(positions)} rows")

    print("Calculating defense vs position...")
    def_vs_pos = calculate_defense_vs_position(game_logs, positions)
    def_vs_pos.to_csv(PROJECT_DATA_DIR / "defense_vs_position.csv", index=False)
    print(f"  -> {len(def_vs_pos)} rows")

    # Rebuild engineered_data.parquet with rolling averages so app startup is fast
    try:
        from utils.feature_engineering import add_rolling_averages
        print("Rebuilding engineered_data.parquet with rolling averages...")
        enriched = add_rolling_averages(game_logs)
        enriched["_date"] = pd.to_datetime(enriched["GAME_DATE"], format="%b %d, %Y", errors="coerce")
        enriched.to_parquet(PROJECT_DATA_DIR / "engineered_data.parquet", index=False)
        latest = enriched["_date"].max()
        print(f"  -> parquet written ({len(enriched)} rows, latest game: {latest.date() if pd.notna(latest) else 'unknown'})")
    except Exception as e:
        logger.warning(f"Could not write parquet: {e}")

    print("All pipeline CSVs exported to data/")
