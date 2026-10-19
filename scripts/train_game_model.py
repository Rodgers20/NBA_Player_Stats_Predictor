"""
Legacy command alias for validated NBA game-model tuning.

Run once (or whenever you want to retrain on fresh data):
    python3 scripts/train_game_model.py

Creates:
    data/model_game_xgb.pkl  — XGBoost model for team score prediction

Features used (all computable at prediction time with no leakage):
    roll_ppg_10   : rolling 10-game team PPG (shift-1 to avoid leakage)
    roll_ppg_5    : rolling 5-game team PPG
    roll_wl_10    : rolling 10-game win rate
    is_home       : 1 if home game
    rest_days     : days since last game (clipped 0–10)
    is_b2b        : 1 if back-to-back (rest_days <= 1)
    opp_roll_ppg_10 : opponent rolling 10-game PPG (proxy for opponent offense)
    opp_roll_ppg_5  : opponent rolling 5-game PPG

Target:
    team_pts : actual team score in the game
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure we can import from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


FEATURES = [
    "roll_ppg_10", "roll_ppg_5", "roll_wl_10",
    "is_home", "rest_days", "is_b2b",
    "opp_roll_ppg_10", "opp_roll_ppg_5",
]
TARGET  = "team_pts"
OUT_PKL = PROJECT_ROOT / "data" / "nba" / "model_game_xgb.pkl"


def build_team_game_features(player_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player logs to one row per (team, game) and engineer features.
    All rolling features use shift(1) to prevent data leakage.
    """
    df = player_logs_df.copy()

    # Parse team / opponent / home flag from MATCHUP e.g. "BOS vs. MIA" or "BOS @ MIA"
    df["team"]     = df["MATCHUP"].str.extract(r'^([A-Z]{2,3})\s+(?:vs\.|@)')[0]
    df["opponent"] = df["MATCHUP"].str.extract(r'(?:vs\.|@)\s+([A-Z]{2,3})')[0]
    df["is_home"]  = df["MATCHUP"].str.contains(r"\bvs\.", regex=True, na=False).astype(int)

    # Drop rows we can't use
    df = df.dropna(subset=["team", "opponent", "_date", "PTS"])
    df["_date"] = pd.to_datetime(df["_date"], errors="coerce")
    df = df.dropna(subset=["_date"])

    # Team game totals: sum player PTS per (team, date)
    team_games = (
        df.groupby(["team", "_date", "opponent", "is_home", "WL"])
        .agg(team_pts=("PTS", "sum"))
        .reset_index()
        .sort_values(["team", "_date"])
    )

    # Rolling offense features (shift-1 = use only past data)
    grp = team_games.groupby("team")["team_pts"]
    team_games["roll_ppg_10"] = grp.transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    team_games["roll_ppg_5"] = grp.transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )

    # Rolling win rate
    wl_num = team_games["WL"].map({"W": 1, "L": 0}).astype(float)
    team_games["_wl_num"] = wl_num
    team_games["roll_wl_10"] = team_games.groupby("team")["_wl_num"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    # Rest days
    team_games["rest_days"] = (
        team_games.groupby("team")["_date"]
        .diff()
        .dt.days
        .fillna(3)
        .clip(0, 10)
    )
    team_games["is_b2b"] = (team_games["rest_days"] <= 1).astype(int)

    # Merge opponent rolling PPG as defensive context
    opp_lookup = team_games[["team", "_date", "roll_ppg_10", "roll_ppg_5"]].copy()
    opp_lookup.columns = ["opponent", "_date", "opp_roll_ppg_10", "opp_roll_ppg_5"]
    team_games = team_games.merge(opp_lookup, on=["opponent", "_date"], how="left")

    # Drop rows with insufficient history
    team_games = team_games.dropna(subset=["roll_ppg_10", "opp_roll_ppg_10", "team_pts"])

    return team_games


def train():
    """Use the date-grouped candidate evaluation and guarded artifact pipeline."""
    from scripts.tune_game_models import run
    run("nba")


if __name__ == "__main__":
    train()
