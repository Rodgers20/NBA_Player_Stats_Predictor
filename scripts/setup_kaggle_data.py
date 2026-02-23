#!/usr/bin/env python3
"""
First-time setup for Kaggle data integration.

1. Validates Kaggle API credentials
2. Downloads the dataset to kagglehub cache
3. Exports pipeline-ready CSVs to data/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_kaggle_credentials():
    """Check if Kaggle API token is configured."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        print("[OK] Kaggle credentials found at ~/.kaggle/kaggle.json")
        return True

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("[OK] Kaggle credentials found in environment variables")
        return True

    print("[ERROR] Kaggle API token not found!")
    print("  Option 1: Place kaggle.json at ~/.kaggle/kaggle.json")
    print("  Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY env vars")
    print("  Get your token: https://www.kaggle.com/settings -> API -> Create New Token")
    return False


def main():
    if not check_kaggle_credentials():
        sys.exit(1)

    from utils.kaggle_loader import download_dataset, export_pipeline_csvs

    print("\nDownloading Kaggle NBA dataset (~876 MB)...")
    path = download_dataset(force=True)
    print(f"Dataset cached at: {path}")

    print("\nExporting pipeline-ready CSVs to data/...")
    export_pipeline_csvs(num_seasons=3)

    print("\nSetup complete!")
    print("Next steps:")
    print("  1. python scripts/train_improved_models.py  (retrain models)")
    print("  2. python dashboard/app.py                   (start dashboard)")


if __name__ == "__main__":
    main()
