"""
One-time discovery script: Download Kaggle dataset and print schemas.
Run this FIRST to see exact column names for building mappings.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kagglehub
import pandas as pd

DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
OUTPUT_DIR = os.path.expanduser("~/.nba_data")

print(f"Downloading dataset to {OUTPUT_DIR}...")
path = kagglehub.dataset_download(DATASET)
print(f"Dataset path: {path}")

# List all files
print(f"\nFiles in dataset directory:")
for f in sorted(os.listdir(path)):
    size_mb = os.path.getsize(os.path.join(path, f)) / (1024 * 1024)
    print(f"  {f} ({size_mb:.1f} MB)")

# Inspect each CSV
csv_files = [
    "PlayerStatistics.csv",
    "TeamStatistics.csv",
    "Games.csv",
    "Players.csv",
    "LeagueSchedule24_25.csv",
    "TeamHistories.csv",
]

for csv_file in csv_files:
    filepath = os.path.join(path, csv_file)
    if not os.path.exists(filepath):
        print(f"\n{'='*70}")
        print(f"FILE: {csv_file} — NOT FOUND")
        continue

    df = pd.read_csv(filepath, nrows=5)
    print(f"\n{'='*70}")
    print(f"FILE: {csv_file}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"\nDtypes:")
    print(df.dtypes.to_string())
    print(f"\nSample (first 3 rows):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.head(3).to_string())
