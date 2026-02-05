#!/usr/bin/env python3
# scripts/collect_sample_data.py
"""
Sample Data Collection - Quick version for testing
==================================================
Collects data for ~20 star players to test our pipeline.
Run the full collection later when you're ready to train the final model.

HOW TO RUN:
    source env/bin/activate
    python scripts/collect_sample_data.py
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.data_fetch import (
    get_player_stats,
    get_team_data,
    get_team_defensive_stats,
)


def get_current_nba_season():
    """
    Get the current NBA season string (e.g., '2025-26').
    NBA season starts in October.
    """
    today = datetime.now()
    year = today.year
    month = today.month

    if month >= 10:  # October or later
        start_year = year
    else:  # January - September
        start_year = year - 1

    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def get_seasons_to_collect(num_seasons=2):
    """Get list of seasons to collect (current + previous)."""
    current = get_current_nba_season()
    start_year = int(current.split("-")[0])

    seasons = []
    for i in range(num_seasons):
        year = start_year - i
        seasons.append(f"{year}-{str(year + 1)[-2:]}")

    return seasons


# =============================================================================
# CONFIGURATION
# =============================================================================

# Sample of star players to collect (diverse positions and teams)
SAMPLE_PLAYERS = [
    # Guards
    "Stephen Curry",
    "Luka Doncic",
    "Shai Gilgeous-Alexander",
    "Trae Young",
    "Tyrese Haliburton",
    # Forwards
    "LeBron James",
    "Kevin Durant",
    "Jayson Tatum",
    "Giannis Antetokounmpo",
    "Kawhi Leonard",
    # Centers
    "Nikola Jokic",
    "Joel Embiid",
    "Anthony Davis",
    "Bam Adebayo",
    "Domantas Sabonis",
    # Rising Stars
    "Anthony Edwards",
    "Ja Morant",
    "Victor Wembanyama",
    "Paolo Banchero",
    "Chet Holmgren",
]

# Dynamically get current and previous season
SEASONS = get_seasons_to_collect(num_seasons=2)
print(f"Collecting seasons: {SEASONS}")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
API_DELAY = 0.6


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def save_csv(df: pd.DataFrame, filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filename} ({len(df)} rows)")


def collect_team_data():
    """Collect team stats for context."""
    print("\n📊 Collecting team data...")

    team_stats = []
    defensive_stats = []

    for season in SEASONS:
        print(f"  {season}...", end=" ", flush=True)

        team_df = get_team_data(season)
        if not team_df.empty:
            team_df["SEASON"] = season
            team_stats.append(team_df)

        def_df = get_team_defensive_stats(season)
        if not def_df.empty:
            def_df["SEASON"] = season
            defensive_stats.append(def_df)

        print("done")
        time.sleep(API_DELAY)

    if team_stats:
        save_csv(pd.concat(team_stats, ignore_index=True), "team_stats.csv")

    if defensive_stats:
        save_csv(pd.concat(defensive_stats, ignore_index=True), "team_defensive_stats.csv")


def collect_player_data():
    """Collect game logs for sample players."""
    print(f"\n🏀 Collecting data for {len(SAMPLE_PLAYERS)} players...")

    all_games = []

    for i, player in enumerate(SAMPLE_PLAYERS, 1):
        print(f"\n[{i}/{len(SAMPLE_PLAYERS)}] {player}")

        for season in SEASONS:
            print(f"  {season}...", end=" ", flush=True)

            df = get_player_stats(player, season)

            if not df.empty:
                df["SEASON"] = season
                all_games.append(df)
                print(f"{len(df)} games")
            else:
                print("no data")

            time.sleep(API_DELAY)

    if all_games:
        combined = pd.concat(all_games, ignore_index=True)
        save_csv(combined, "player_game_logs.csv")
        return combined

    return pd.DataFrame()


def main():
    print("=" * 50)
    print("NBA SAMPLE DATA COLLECTION")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)

    ensure_data_dir()

    # Collect team data first (fast)
    collect_team_data()

    # Collect player game logs
    df = collect_player_data()

    # Summary
    print("\n" + "=" * 50)
    print("COLLECTION COMPLETE!")
    print("=" * 50)

    if not df.empty:
        print(f"\n📈 Total game logs: {len(df):,}")
        print(f"👥 Players: {df['PLAYER_NAME'].nunique()}")
        print(f"📅 Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

        print("\n📁 Files saved to data/:")
        for f in os.listdir(DATA_DIR):
            if f.endswith(".csv"):
                size = os.path.getsize(os.path.join(DATA_DIR, f)) / 1024
                print(f"  • {f} ({size:.1f} KB)")

        print("\nSample data preview:")
        print(df[["PLAYER_NAME", "GAME_DATE", "MATCHUP", "PTS", "AST", "REB"]].head())

    print(f"\nFinished: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
