#!/usr/bin/env python3
"""Collect remaining players using existing player list (no heavy API calls)."""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetch import get_player_stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEASONS = ["2025-26", "2024-25", "2023-24"]
API_DELAY = 1.5

def load_collected_players():
    """Load players already collected."""
    filepath = os.path.join(DATA_DIR, "collection_progress.txt")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_player_progress(player_name):
    """Mark player as collected."""
    filepath = os.path.join(DATA_DIR, "collection_progress.txt")
    with open(filepath, "a") as f:
        f.write(f"{player_name}\n")

def main():
    # Load player list from existing positions file
    positions_file = os.path.join(DATA_DIR, "player_positions.csv")
    if not os.path.exists(positions_file):
        print("ERROR: player_positions.csv not found!")
        return

    positions_df = pd.read_csv(positions_file)
    all_players = positions_df["PLAYER_NAME"].unique().tolist()
    print(f"Total players in database: {len(all_players)}")

    # Check which are already collected
    collected = load_collected_players()
    remaining = [p for p in all_players if p not in collected]

    print(f"Already collected: {len(collected)}")
    print(f"Remaining to collect: {len(remaining)}")

    if not remaining:
        print("\nAll players already collected!")
        return

    print(f"\nWill collect {len(remaining)} players across {len(SEASONS)} seasons")
    print(f"Estimated time: {len(remaining) * len(SEASONS) * API_DELAY / 60:.1f} minutes")

    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    # Load existing game logs to append to
    logs_file = os.path.join(DATA_DIR, "player_game_logs.csv")
    if os.path.exists(logs_file):
        existing_logs = pd.read_csv(logs_file)
        all_game_logs = [existing_logs]
        print(f"\nLoaded {len(existing_logs)} existing game logs")
    else:
        all_game_logs = []

    # Collect remaining players
    for i, player_name in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] {player_name}")

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

            time.sleep(API_DELAY)

        if player_games:
            player_df = pd.concat(player_games, ignore_index=True)
            all_game_logs.append(player_df)
            save_player_progress(player_name)

        # Save checkpoint every 25 players
        if i % 25 == 0 and all_game_logs:
            print(f"\n>>> Checkpoint: Saving progress ({i} players)...")
            checkpoint_df = pd.concat(all_game_logs, ignore_index=True)
            checkpoint_df.to_csv(logs_file, index=False)
            print(f"    Saved {len(checkpoint_df)} total game logs")

    # Final save
    if all_game_logs:
        final_df = pd.concat(all_game_logs, ignore_index=True)
        final_df.to_csv(logs_file, index=False)
        print(f"\n{'='*60}")
        print("COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total game logs: {len(final_df)}")
        print(f"Unique players: {final_df['PLAYER_NAME'].nunique()}")

if __name__ == "__main__":
    main()
