# utils/feature_engineering.py
"""
Feature Engineering Module
==========================
This module transforms raw game data into ML-ready features.

WHAT IS FEATURE ENGINEERING?
----------------------------
Raw data: "LeBron scored 30 points on Jan 15 vs Lakers"
ML needs: Numbers that capture patterns

We create features like:
- rolling_avg_pts_5: Average points in last 5 games (recent form)
- is_home: 1 if home game, 0 if away
- days_rest: Days since last game (fatigue factor)
- opp_def_rating: How good is the opponent's defense?

WHY THESE FEATURES?
-------------------
1. ROLLING AVERAGES: A player's last 5 games are more predictive than
   their season average. If LeBron averages 25 PPG but scored 35, 33, 30
   in his last 3, he's hot - predict higher!

2. HOME/AWAY: Most players score more at home (crowd energy, no travel).
   The model can learn this pattern.

3. REST DAYS: Back-to-back games = tired legs = fewer points.
   Players with 2+ days rest often perform better.

4. OPPONENT DEFENSE: If facing the #1 defense, expect lower stats.
   If facing the #30 defense, expect higher stats.

5. MINUTES TREND: If a player's minutes are increasing (maybe a teammate
   got injured), their stats will likely increase too.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


# =============================================================================
# DATE PARSING
# =============================================================================

def parse_game_date(date_str: str) -> Optional[datetime]:
    """
    Parse NBA game date strings into datetime objects.

    WHY: The NBA API returns dates in various formats like "Jan 15, 2025"
    We need datetime objects to calculate days between games.

    Args:
        date_str: Date string like "Jan 15, 2025" or "2025-01-15"

    Returns:
        datetime object or None if parsing fails
    """
    formats = [
        "%b %d, %Y",   # "Jan 15, 2025"
        "%Y-%m-%d",    # "2025-01-15"
        "%m/%d/%Y",    # "01/15/2025"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


# =============================================================================
# ROLLING AVERAGES
# =============================================================================

def add_rolling_averages(
    df: pd.DataFrame,
    stats: list[str] = None,
    windows: list[int] = None
) -> pd.DataFrame:
    """
    Add rolling average features for specified stats.

    HOW ROLLING AVERAGES WORK:
    For each game, we look back at the previous N games and calculate
    the average. This captures "recent form".

    Example with window=3:
        Game 1: 20 pts → rolling_avg = NaN (not enough history)
        Game 2: 25 pts → rolling_avg = NaN
        Game 3: 30 pts → rolling_avg = NaN
        Game 4: 22 pts → rolling_avg = (20+25+30)/3 = 25.0
        Game 5: 28 pts → rolling_avg = (25+30+22)/3 = 25.7

    IMPORTANT: We use shift(1) to avoid "data leakage" - the model
    shouldn't see the current game's stats when predicting!

    Args:
        df: DataFrame with game logs (must have PLAYER_NAME column)
        stats: Which columns to calculate rolling averages for
               Default: ["PTS", "AST", "REB", "MIN"]
        windows: Rolling window sizes. Default: [5, 10, 20]

    Returns:
        DataFrame with new rolling average columns added
    """
    if stats is None:
        stats = ["PTS", "AST", "REB", "MIN", "FGA", "FG_PCT", "FG3A", "FG3_PCT"]

    if windows is None:
        windows = [5, 10, 20]

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Sort by player and date (oldest first) for correct rolling calculation
    df["_date_parsed"] = df["GAME_DATE"].apply(parse_game_date)
    df = df.sort_values(["PLAYER_NAME", "_date_parsed"])

    # Calculate rolling averages for each player separately
    for stat in stats:
        if stat not in df.columns:
            continue

        for window in windows:
            col_name = f"rolling_avg_{stat.lower()}_{window}"

            # groupby ensures we only look at each player's own games
            # shift(1) ensures we don't include the current game
            df[col_name] = (
                df.groupby("PLAYER_NAME")[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )

    # Clean up
    df = df.drop("_date_parsed", axis=1)

    return df


# =============================================================================
# HOME/AWAY FEATURE
# =============================================================================

def add_home_away_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary feature indicating home (1) or away (0) games.

    HOW TO DETECT HOME/AWAY:
    The MATCHUP column shows: "LAL vs. GSW" (home) or "LAL @ GSW" (away)
    - "vs." = home game (we're hosting)
    - "@" = away game (we're visiting)

    WHY THIS MATTERS:
    Players typically perform better at home:
    - Familiar arena, crowd support
    - No travel fatigue
    - Sleep in own bed

    Example data shows LeBron: 24.9 PPG at home vs 24.0 PPG away
    """
    df = df.copy()

    # Check for "vs." in matchup string
    # "LAL vs. GSW" → home=1, "LAL @ GSW" → home=0
    df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)

    return df


# =============================================================================
# REST DAYS FEATURE
# =============================================================================

def add_rest_days_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature for days of rest since last game.

    WHY REST MATTERS:
    - 0 days (back-to-back): Player is tired, expect lower stats
    - 1 day: Normal rest
    - 2+ days: Well rested, potentially better performance
    - 7+ days: Coming off injury/All-Star break, might be rusty

    The model can learn these patterns automatically.
    """
    df = df.copy()

    # Parse dates
    df["_date_parsed"] = df["GAME_DATE"].apply(parse_game_date)

    # Sort by player and date
    df = df.sort_values(["PLAYER_NAME", "_date_parsed"])

    # Calculate days since last game for each player
    df["days_rest"] = (
        df.groupby("PLAYER_NAME")["_date_parsed"]
        .diff()  # Difference from previous row
        .dt.days  # Convert timedelta to days
        .fillna(3)  # First game of season = assume normal rest
    )

    # Cap at reasonable values (if > 30 days, probably season start or injury)
    df["days_rest"] = df["days_rest"].clip(0, 14)

    # Create binary feature for back-to-back games
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)

    df = df.drop("_date_parsed", axis=1)

    return df


# =============================================================================
# OPPONENT FEATURES
# =============================================================================

def extract_opponent(matchup: str) -> str:
    """
    Extract opponent team abbreviation from matchup string.

    Examples:
        "LAL vs. GSW" → "GSW" (home game vs Golden State)
        "LAL @ PHX" → "PHX" (away game at Phoenix)
    """
    if "@" in matchup:
        # Away game: "LAL @ PHX" → opponent is after @
        return matchup.split("@")[1].strip()
    elif "vs." in matchup:
        # Home game: "LAL vs. GSW" → opponent is after vs.
        return matchup.split("vs.")[1].strip()
    return ""


def add_opponent_features(
    df: pd.DataFrame,
    team_defensive_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Add opponent-related features like defensive rating.

    WHY OPPONENT DEFENSE MATTERS:
    If you're playing against the #1 defense (e.g., Boston Celtics),
    expect your stats to be lower than against the #30 defense.

    We merge in the opponent's defensive stats:
    - OPP_PTS: How many points they allow (higher = worse defense)
    - OPP_FG_PCT: Opponent field goal % (higher = worse defense)

    This lets the model adjust predictions based on matchup difficulty.
    """
    df = df.copy()

    # Extract opponent from matchup
    df["opponent"] = df["MATCHUP"].apply(extract_opponent)

    # Get team abbreviations mapping
    # We need to match opponent name to team stats
    if "TEAM_ABBREVIATION" in team_defensive_stats.columns:
        # Create lookup for defensive stats by team and season
        def_stats = team_defensive_stats.copy()

        # Select relevant columns and rename for clarity
        def_cols = ["TEAM_ABBREVIATION", "SEASON", "OPP_PTS", "OPP_FG_PCT", "OPP_FG3_PCT"]
        available_cols = [c for c in def_cols if c in def_stats.columns]

        if len(available_cols) >= 3:
            def_lookup = def_stats[available_cols].copy()

            # Rename columns to indicate these are opponent stats
            def_lookup = def_lookup.rename(columns={
                "OPP_PTS": "opp_def_pts_allowed",
                "OPP_FG_PCT": "opp_def_fg_pct",
                "OPP_FG3_PCT": "opp_def_fg3_pct",
            })

            # Merge on opponent and season
            df = df.merge(
                def_lookup,
                left_on=["opponent", "SEASON"],
                right_on=["TEAM_ABBREVIATION", "SEASON"],
                how="left"
            )

            # Clean up
            if "TEAM_ABBREVIATION" in df.columns:
                df = df.drop("TEAM_ABBREVIATION", axis=1)

    return df


# =============================================================================
# MINUTES TREND FEATURE
# =============================================================================

def add_minutes_trend(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add feature for recent minutes trend.

    WHY MINUTES TREND MATTERS:
    If a player's minutes are increasing (maybe a teammate got injured
    or they're playing better), their counting stats will likely increase.

    We calculate the change in average minutes over recent games.
    - Positive trend: Minutes increasing → expect more stats
    - Negative trend: Minutes decreasing → expect fewer stats
    """
    df = df.copy()

    # Sort by player and date
    df["_date_parsed"] = df["GAME_DATE"].apply(parse_game_date)
    df = df.sort_values(["PLAYER_NAME", "_date_parsed"])

    # Calculate rolling average minutes
    df["_rolling_min"] = (
        df.groupby("PLAYER_NAME")["MIN"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    # Calculate trend (current rolling avg - previous rolling avg)
    df["minutes_trend"] = (
        df.groupby("PLAYER_NAME")["_rolling_min"]
        .diff()
        .fillna(0)
    )

    df = df.drop(["_date_parsed", "_rolling_min"], axis=1)

    return df


# =============================================================================
# SEASON AVERAGES (for comparison)
# =============================================================================

def add_season_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season-to-date averages as baseline features.

    WHY: Rolling averages capture recent form, but season averages
    provide a baseline. If a player averages 25 PPG for the season
    but only 20 PPG in the last 5 games, they might bounce back.

    The model can learn to balance recent form vs. baseline.
    """
    df = df.copy()

    # Parse dates and sort
    df["_date_parsed"] = df["GAME_DATE"].apply(parse_game_date)
    df = df.sort_values(["PLAYER_NAME", "SEASON", "_date_parsed"])

    # Calculate expanding mean (season average up to each game)
    for stat in ["PTS", "AST", "REB", "MIN"]:
        if stat in df.columns:
            df[f"season_avg_{stat.lower()}"] = (
                df.groupby(["PLAYER_NAME", "SEASON"])[stat]
                .transform(lambda x: x.shift(1).expanding().mean())
            )

    df = df.drop("_date_parsed", axis=1)

    return df


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def engineer_features(
    game_logs: pd.DataFrame,
    team_defensive_stats: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Apply all feature engineering steps to create ML-ready data.

    This is the MAIN FUNCTION to call. It applies all transformations:
    1. Rolling averages (5, 10, 20 games)
    2. Home/away indicator
    3. Rest days calculation
    4. Opponent defensive features (if team stats provided)
    5. Minutes trend
    6. Season averages

    Args:
        game_logs: DataFrame with raw game logs
        team_defensive_stats: Optional DataFrame with team defensive stats

    Returns:
        DataFrame with all features added, ready for ML

    Example:
        >>> from utils.feature_engineering import engineer_features
        >>> df = pd.read_csv("data/player_game_logs.csv")
        >>> team_def = pd.read_csv("data/team_defensive_stats.csv")
        >>> features = engineer_features(df, team_def)
        >>> features.columns
        ['PLAYER_NAME', 'PTS', 'rolling_avg_pts_5', 'rolling_avg_pts_10',
         'is_home', 'days_rest', 'opp_def_pts_allowed', ...]
    """
    print("Engineering features...")

    # Step 1: Rolling averages
    print("  - Adding rolling averages...")
    df = add_rolling_averages(game_logs)

    # Step 2: Home/away
    print("  - Adding home/away indicator...")
    df = add_home_away_feature(df)

    # Step 3: Rest days
    print("  - Calculating rest days...")
    df = add_rest_days_feature(df)

    # Step 4: Opponent features (if team stats provided)
    if team_defensive_stats is not None:
        print("  - Adding opponent defensive features...")
        df = add_opponent_features(df, team_defensive_stats)

    # Step 5: Minutes trend
    print("  - Calculating minutes trend...")
    df = add_minutes_trend(df)

    # Step 6: Season averages
    print("  - Adding season averages...")
    df = add_season_averages(df)

    print(f"  ✓ Created {len(df.columns)} total columns")

    return df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of feature columns (excluding target and identifier columns).

    Use this to know which columns to feed to the ML model.
    """
    exclude = [
        "PLAYER_NAME", "GAME_DATE", "MATCHUP", "WL", "SEASON",
        "Player_ID", "Game_ID", "SEASON_ID",
        "PTS", "AST", "REB",  # These are targets, not features
        "opponent", "VIDEO_AVAILABLE"
    ]

    features = [col for col in df.columns if col not in exclude]

    # Only include numeric columns
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()

    return numeric_features


def prepare_training_data(
    df: pd.DataFrame,
    target: str = "PTS",
    features: list[str] = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training by selecting features and target.

    Args:
        df: DataFrame with engineered features
        target: Column to predict (e.g., "PTS", "AST", "REB")
        features: List of feature columns (if None, auto-detect)

    Returns:
        Tuple of (X, y) where X is features DataFrame, y is target Series
    """
    if features is None:
        features = get_feature_columns(df)

    # Drop rows with missing values in features or target
    required_cols = features + [target]
    df_clean = df.dropna(subset=required_cols)

    X = df_clean[features]
    y = df_clean[target]

    print(f"Training data prepared:")
    print(f"  - Samples: {len(X)}")
    print(f"  - Features: {len(features)}")
    print(f"  - Target: {target}")

    return X, y
