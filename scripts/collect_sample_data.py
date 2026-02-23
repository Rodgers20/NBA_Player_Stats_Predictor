#!/usr/bin/env python3
# scripts/collect_sample_data.py
"""
Sample Data Collection (Kaggle Edition)
========================================
Downloads NBA data from Kaggle and exports pipeline-ready CSVs.
This is a lighter version of collect_training_data.py (1 season only).

HOW TO RUN:
    cd NBA_Player_Stats_Predictor
    python scripts/collect_sample_data.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kaggle_loader import download_dataset, export_pipeline_csvs


def main():
    print("=" * 50)
    print("NBA SAMPLE DATA COLLECTION (Kaggle)")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)

    # Download latest dataset
    print("\nDownloading latest Kaggle dataset...")
    path = download_dataset(force=False)
    print(f"  Dataset at: {path}")

    # Export 1 season of data (quick)
    print("\nExporting 1 season of data...")
    export_pipeline_csvs(num_seasons=1)

    print(f"\nFinished: {datetime.now().strftime('%H:%M:%S')}")
    print("\nNext: python scripts/train_improved_models.py")


if __name__ == "__main__":
    main()
