#!/usr/bin/env python3
# scripts/collect_training_data.py
"""
Training Data Collection Script
===============================
This script fetches historical NBA data and saves it locally for ML training.

WHY WE NEED THIS:
-----------------
1. API calls are slow (0.6s delay to avoid rate limiting)
2. Fetching 300 players × 3 seasons each time is inefficient
3. We want reproducible training data
4. Working with local files is much faster than live API calls

WHAT IT COLLECTS:
-----------------
1. Player game logs (3 seasons) - The main training data
2. Team stats - For context features
3. Team defensive ratings - For matchup predictions

HOW TO RUN:
-----------
    cd NBA_Player_Stats_Predictor
    source env/bin/activate
    python scripts/collect_training_data.py

The script will show progress and save files to the data/ directory.
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path so we can import our utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.data_fetch import (
    get_player_stats,
    get_all_players_season_stats,
    get_team_data,
    get_team_defensive_stats,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Seasons to collect (most recent first)
SEASONS = ["2024-25", "2023-24", "2022-23"]

# Minimum games played to include a player (filters out bench warmers)
MIN_GAMES = 20

# Where to save the data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Delay between API calls (to avoid rate limiting)
API_DELAY = 0.6  # seconds


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")


def save_dataframe(df: pd.DataFrame, filename: str):
    """Save a DataFrame to CSV in the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath} ({len(df)} rows)")


def load_existing_players() -> set:
    """
    Load list of players we've already collected data for.
    This allows us to resume interrupted collection.
    """
    filepath = os.path.join(DATA_DIR, "collection_progress.txt")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return set(line.strip() for line in f)
    return set()


def save_player_progress(player_name: str):
    """Record that we've collected data for a player."""
    filepath = os.path.join(DATA_DIR, "collection_progress.txt")
    with open(filepath, "a") as f:
        f.write(f"{player_name}\n")


# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def collect_team_data():
    """
    Collect team stats for all seasons.

    WHY: Team context affects individual predictions.
    Fast-paced teams = more possessions = higher counting stats.
    """
    print("\n" + "=" * 60)
    print("COLLECTING TEAM DATA")
    print("=" * 60)

    all_team_stats = []
    all_defensive_stats = []

    for season in SEASONS:
        print(f"\nFetching team stats for {season}...")

        # Offensive/general stats
        team_df = get_team_data(season)
        if not team_df.empty:
            team_df["SEASON"] = season
            all_team_stats.append(team_df)
        time.sleep(API_DELAY)

        # Defensive stats
        def_df = get_team_defensive_stats(season)
        if not def_df.empty:
            def_df["SEASON"] = season
            all_defensive_stats.append(def_df)
        time.sleep(API_DELAY)

    # Combine and save
    if all_team_stats:
        combined_teams = pd.concat(all_team_stats, ignore_index=True)
        save_dataframe(combined_teams, "team_stats.csv")

    if all_defensive_stats:
        combined_defense = pd.concat(all_defensive_stats, ignore_index=True)
        save_dataframe(combined_defense, "team_defensive_stats.csv")


def get_players_to_collect() -> list[str]:
    """
    Get list of players to collect data for.

    STRATEGY:
    1. Get all players from the current season
    2. Filter to those with MIN_GAMES played
    3. Skip any we've already collected (for resume support)
    """
    print("\n" + "=" * 60)
    print("IDENTIFYING PLAYERS TO COLLECT")
    print("=" * 60)

    # Get current season players
    print(f"\nFetching player list for {SEASONS[0]}...")
    current_players = get_all_players_season_stats(SEASONS[0])

    if current_players.empty:
        print("ERROR: Could not fetch player list!")
        return []

    # Filter by games played
    active_players = current_players[current_players["GP"] >= MIN_GAMES]
    player_names = active_players["PLAYER_NAME"].tolist()

    print(f"Found {len(player_names)} players with {MIN_GAMES}+ games")

    # Check for already collected (resume support)
    already_collected = load_existing_players()
    if already_collected:
        player_names = [p for p in player_names if p not in already_collected]
        print(f"Resuming collection - {len(already_collected)} already done")
        print(f"Remaining: {len(player_names)} players")

    return player_names


def collect_player_game_logs(player_names: list[str]):
    """
    Collect game-by-game data for all players across all seasons.

    THIS IS THE MAIN TRAINING DATA.
    Each row = one game played by one player.

    Example output:
        PLAYER_NAME    GAME_DATE    MATCHUP       PTS   AST   REB   SEASON
        LeBron James   2024-10-22   LAL vs. MIN   25    8     7     2024-25
        LeBron James   2024-10-24   LAL @ PHX     30    10    5     2024-25
        ...
    """
    print("\n" + "=" * 60)
    print("COLLECTING PLAYER GAME LOGS")
    print("=" * 60)

    all_game_logs = []
    total_players = len(player_names)

    for i, player_name in enumerate(player_names, 1):
        print(f"\n[{i}/{total_players}] {player_name}")

        player_games = []

        for season in SEASONS:
            print(f"  - {season}...", end=" ", flush=True)

            df = get_player_stats(player_name, season)

            if not df.empty:
                df["SEASON"] = season
                player_games.append(df)
                print(f"{len(df)} games")
            else:
                print("no data")

            # Respect rate limits
            time.sleep(API_DELAY)

        # Combine this player's games
        if player_games:
            player_df = pd.concat(player_games, ignore_index=True)
            all_game_logs.append(player_df)
            save_player_progress(player_name)

        # Save incrementally every 50 players (in case of crash)
        if i % 50 == 0 and all_game_logs:
            print(f"\n>>> Checkpoint: Saving progress ({i} players)...")
            checkpoint_df = pd.concat(all_game_logs, ignore_index=True)
            save_dataframe(checkpoint_df, "player_game_logs_checkpoint.csv")

    # Final save
    if all_game_logs:
        final_df = pd.concat(all_game_logs, ignore_index=True)
        save_dataframe(final_df, "player_game_logs.csv")

        # Remove checkpoint file
        checkpoint_path = os.path.join(DATA_DIR, "player_game_logs_checkpoint.csv")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return final_df

    return pd.DataFrame()


def print_summary(game_logs_df: pd.DataFrame):
    """Print a summary of the collected data."""
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE!")
    print("=" * 60)

    if game_logs_df.empty:
        print("No data collected.")
        return

    print(f"\nTotal game logs: {len(game_logs_df):,}")
    print(f"Unique players: {game_logs_df['PLAYER_NAME'].nunique()}")
    print(f"Seasons covered: {game_logs_df['SEASON'].unique().tolist()}")

    print("\nSample data:")
    print(game_logs_df[["PLAYER_NAME", "GAME_DATE", "MATCHUP", "PTS", "AST", "REB", "SEASON"]].head(5))

    print("\nFiles saved to:", DATA_DIR)
    for f in os.listdir(DATA_DIR):
        if f.endswith(".csv"):
            filepath = os.path.join(DATA_DIR, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main entry point for data collection.

    EXECUTION ORDER:
    1. Team data first (fast, few API calls)
    2. Player list (1 API call)
    3. Player game logs (slow, many API calls)
    """
    print("=" * 60)
    print("NBA TRAINING DATA COLLECTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seasons: {SEASONS}")
    print("=" * 60)

    # Setup
    ensure_data_dir()

    # Step 1: Team data
    collect_team_data()

    # Step 2: Get player list
    player_names = get_players_to_collect()

    if not player_names:
        print("\nNo players to collect. Exiting.")
        return

    # Step 3: Collect player game logs
    # NOTE: This can take 30+ minutes for 300 players
    print(f"\nThis will fetch data for {len(player_names)} players across {len(SEASONS)} seasons.")
    print(f"Estimated time: {len(player_names) * len(SEASONS) * API_DELAY / 60:.1f} minutes")

    response = input("\nProceed? (y/n): ").strip().lower()
    if response != "y":
        print("Aborted.")
        return

    game_logs = collect_player_game_logs(player_names)

    # Summary
    print_summary(game_logs)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
