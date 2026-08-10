#!/usr/bin/env python3
"""First-time WNBA data setup.

Fetches WNBA player + team logs via nba_api and writes CSVs to data/wnba/.
No Kaggle credentials needed (unlike NBA).

Usage:
    python scripts/setup_wnba_data.py
    python scripts/setup_wnba_data.py --seasons 2023,2024,2025
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wnba_loader import export_pipeline_csvs, get_recent_wnba_seasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated seasons (e.g. 2023,2024,2025). Default: last 3.",
    )
    args = parser.parse_args()

    seasons = args.seasons.split(",") if args.seasons else get_recent_wnba_seasons(3)
    print(f"Exporting WNBA seasons: {seasons}")

    export_pipeline_csvs(seasons)
    print("\nDone. Next: run scripts/train_improved_models.py --league wnba (Phase 1 Step 4)")


if __name__ == "__main__":
    main()
