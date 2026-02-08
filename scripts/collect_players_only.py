#!/usr/bin/env python3
"""Quick script to collect remaining players (skips team data)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.collect_training_data import (
    get_players_to_collect,
    collect_player_game_logs,
    print_summary
)

print("=" * 60)
print("COLLECTING REMAINING PLAYERS (skipping team data)")
print("=" * 60)

players = get_players_to_collect()

if not players:
    print("No players left to collect!")
else:
    print(f"\nWill collect {len(players)} remaining players")
    response = input("Proceed? (y/n): ").strip().lower()

    if response == 'y':
        game_logs = collect_player_game_logs(players)
        print_summary(game_logs)
    else:
        print("Aborted.")
