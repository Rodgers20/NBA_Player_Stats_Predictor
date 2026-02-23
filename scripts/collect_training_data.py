#!/usr/bin/env python3
# scripts/collect_training_data.py
"""
Training Data Collection Script (Kaggle Edition)
=================================================
Downloads NBA data from Kaggle and exports pipeline-ready CSVs.

Much faster than the old API approach:
- Old: 300 players x 3 seasons x 1.5s delay = 22+ minutes
- New: Single bulk download + transform = ~2 minutes

HOW TO RUN:
    cd NBA_Player_Stats_Predictor
    python scripts/collect_training_data.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kaggle_loader import download_dataset, export_pipeline_csvs


def main():
    print("=" * 60)
    print("NBA TRAINING DATA COLLECTION (Kaggle)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Download/refresh Kaggle data
    print("\nStep 1: Downloading latest Kaggle dataset...")
    path = download_dataset(force=True)
    print(f"  Dataset at: {path}")

    # Step 2: Export all pipeline CSVs
    print("\nStep 2: Transforming and exporting data...")
    export_pipeline_csvs(num_seasons=3)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext: python scripts/train_improved_models.py")


if __name__ == "__main__":
    main()
