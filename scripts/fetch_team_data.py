#!/usr/bin/env python3
# scripts/fetch_team_data.py
"""
Fetch team data from Kaggle dataset and save to CSV.

HOW TO RUN:
    cd NBA_Player_Stats_Predictor
    python scripts/fetch_team_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kaggle_loader import load_team_stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

team_data = load_team_stats(num_seasons=1)
print(team_data.head())

output_path = os.path.join(DATA_DIR, "team_stats.csv")
team_data.to_csv(output_path, index=False)
print(f"Team data saved to {output_path} ({len(team_data)} rows)")
