#!/usr/bin/env python3
# scripts/train_improved_models.py
"""
Improved Model Training Script
==============================
Train NBA prediction models with enhanced features:
- Position features (Guard/Forward/Center)
- Pace features (team possessions per game)
- Matchup difficulty (elite/good/poor defense tiers)
- Hyperparameter tuning (optional)

USAGE:
    python scripts/train_improved_models.py

    # With hyperparameter tuning (slower but better):
    python scripts/train_improved_models.py --tune

    # Train specific stat only:
    python scripts/train_improved_models.py --stat PTS
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.feature_engineering import engineer_features
from models.predictor import NBAPredictor

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

TARGETS = ["PTS", "AST", "REB"]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load all data files for feature engineering."""
    print("Loading data files...")

    # Required files
    game_logs_path = os.path.join(DATA_DIR, "player_game_logs.csv")
    team_def_path = os.path.join(DATA_DIR, "team_defensive_stats.csv")

    if not os.path.exists(game_logs_path):
        print(f"ERROR: {game_logs_path} not found!")
        print("Run 'python scripts/collect_training_data.py' first.")
        return None

    game_logs = pd.read_csv(game_logs_path)
    team_def = pd.read_csv(team_def_path) if os.path.exists(team_def_path) else pd.DataFrame()

    # Optional enhanced files
    team_stats_path = os.path.join(DATA_DIR, "team_stats.csv")
    positions_path = os.path.join(DATA_DIR, "player_positions.csv")
    def_vs_pos_path = os.path.join(DATA_DIR, "defense_vs_position.csv")

    team_stats = pd.read_csv(team_stats_path) if os.path.exists(team_stats_path) else None
    positions = pd.read_csv(positions_path) if os.path.exists(positions_path) else None
    def_vs_pos = pd.read_csv(def_vs_pos_path) if os.path.exists(def_vs_pos_path) else None

    print(f"  - Game logs: {len(game_logs)} records")
    print(f"  - Team defensive stats: {'loaded' if not team_def.empty else 'not found'}")
    print(f"  - Team stats (pace): {'loaded' if team_stats is not None else 'not found'}")
    print(f"  - Player positions: {'loaded' if positions is not None else 'not found'}")
    print(f"  - Defense vs position: {'loaded' if def_vs_pos is not None else 'not found'}")

    return {
        "game_logs": game_logs,
        "team_def": team_def,
        "team_stats": team_stats,
        "positions": positions,
        "def_vs_pos": def_vs_pos
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    df: pd.DataFrame,
    target: str,
    model_type: str = "xgboost",
    tune: bool = False
) -> dict:
    """
    Train a single model for a target stat.

    Args:
        df: DataFrame with engineered features
        target: Which stat to predict (PTS, AST, REB)
        model_type: Model type (xgboost, random_forest)
        tune: Whether to run hyperparameter tuning

    Returns:
        Dictionary with metrics and model path
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {target} MODEL")
    print(f"{'='*60}")

    predictor = NBAPredictor(model_type=model_type)

    if tune:
        print("Running hyperparameter tuning...")
        best_params = predictor.tune_hyperparameters(df, target=target)
        print(f"Best params: {best_params}")
    else:
        metrics = predictor.train(df, target=target)

    # Get feature importance
    importance_df = predictor.get_feature_importance_df()
    if not importance_df.empty:
        print("\nTop 10 Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{target.lower()}_predictor.pkl")
    predictor.save(model_path)

    return {
        "target": target,
        "metrics": predictor.metrics,
        "model_path": model_path
    }


def train_all_models(df: pd.DataFrame, tune: bool = False) -> list:
    """Train models for all target stats."""
    results = []

    for target in TARGETS:
        result = train_model(df, target, tune=tune)
        results.append(result)

    return results


def print_summary(results: list):
    """Print training summary."""
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    print("\nModel Performance:")
    print("-" * 40)
    print(f"{'Target':<10} {'MAE':>10} {'R²':>10}")
    print("-" * 40)

    for result in results:
        target = result["target"]
        mae = result["metrics"].get("mae", "N/A")
        r2 = result["metrics"].get("r2", "N/A")
        print(f"{target:<10} {mae:>10} {r2:>10}")

    print("-" * 40)
    print(f"\nModels saved to: {MODELS_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train NBA prediction models")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--stat", type=str, help="Train specific stat only (PTS, AST, REB)")
    parser.add_argument("--model", type=str, default="xgboost",
                       choices=["xgboost", "random_forest"],
                       help="Model type to use")
    args = parser.parse_args()

    print("=" * 60)
    print("NBA IMPROVED MODEL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load data
    data = load_all_data()
    if data is None:
        return

    # Engineer features
    print("\nEngineering features...")
    df = engineer_features(
        data["game_logs"],
        data["team_def"],
        team_stats=data["team_stats"],
        player_positions=data["positions"],
        defense_vs_position=data["def_vs_pos"]
    )

    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")

    # Train models
    if args.stat:
        if args.stat.upper() not in TARGETS:
            print(f"ERROR: Invalid stat '{args.stat}'. Must be one of: {TARGETS}")
            return
        results = [train_model(df, args.stat.upper(), args.model, args.tune)]
    else:
        results = train_all_models(df, tune=args.tune)

    # Summary
    print_summary(results)
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
