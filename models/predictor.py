# models/predictor.py
"""
NBA Player Stats Predictor
==========================
This module contains the main ML models for predicting player performance.

MACHINE LEARNING CONCEPTS EXPLAINED:
------------------------------------

1. REGRESSION (predicting numbers):
   We want to predict: "How many points will LeBron score?"
   This is a REGRESSION problem because the answer is a continuous number.

2. FEATURES vs TARGETS:
   - Features (X): The inputs we use to make predictions
     Example: rolling_avg_pts_5, is_home, days_rest, opp_def_rating
   - Target (y): What we're trying to predict
     Example: PTS (actual points scored)

3. TRAINING vs TESTING:
   We split our data 80/20:
   - Training set (80%): Model learns patterns from this data
   - Test set (20%): We check accuracy on data the model hasn't seen

   WHY? If we test on training data, the model could just "memorize"
   the answers instead of learning generalizable patterns.

4. MODEL SELECTION:
   We try multiple models and pick the best one:
   - Random Forest: Creates many decision trees and averages them
   - XGBoost: Builds trees sequentially, each fixing the previous errors
   - Ridge Regression: Linear model with regularization to prevent overfitting

5. EVALUATION METRICS:
   - MAE (Mean Absolute Error): Average difference between predicted and actual
     MAE of 3.5 means predictions are off by 3.5 points on average
   - R² (R-squared): How much variance the model explains (0-1, higher is better)
     R² of 0.7 means model explains 70% of the variance in scoring
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# XGBoost is optional (better performance but not always installed)
# It requires libomp on Mac: brew install libomp
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    # XGBoost may fail to load even if installed (missing libomp)
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available, using Ridge instead")


class StatPredictor:
    """
    Main class for NBA player stats prediction.

    This class handles:
    1. Training models for PTS, AST, REB predictions
    2. Making predictions for new games
    3. Saving/loading trained models
    4. Evaluating model performance

    Example Usage:
    --------------
    >>> from models.predictor import StatPredictor
    >>> from utils.feature_engineering import engineer_features
    >>>
    >>> # Load and prepare data
    >>> df = pd.read_csv("data/player_game_logs.csv")
    >>> features_df = engineer_features(df)
    >>>
    >>> # Train the model
    >>> predictor = StatPredictor()
    >>> predictor.train(features_df, target="PTS")
    >>>
    >>> # Make a prediction
    >>> prediction = predictor.predict(new_game_features)
    >>> print(f"Predicted points: {prediction}")
    """

    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the predictor.

        Args:
            model_type: Which model to use. Options:
                - "xgboost": Best accuracy, handles missing data (recommended)
                - "random_forest": Good accuracy, works out-of-box
                - "gradient_boosting": Similar to XGBoost but slower
                - "ridge": Fast, simple, interpretable

        HOW TO CHOOSE:
        - Start with "random_forest" for quick results
        - Use "xgboost" for best accuracy (competition/production)
        - Use "ridge" if you need to understand which features matter most
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()  # Normalizes features to similar scales
        self.feature_columns = None
        self.target = None
        self.is_trained = False
        self.metrics = {}

    def _create_model(self):
        """
        Create the underlying ML model based on model_type.

        WHAT EACH MODEL DOES:
        - RandomForest: Creates 100 decision trees, each trained on random
          subsets of data. Final prediction = average of all trees.
          This reduces overfitting (single trees memorize; forests generalize).

        - XGBoost: Builds trees sequentially. Each new tree focuses on
          examples the previous trees got wrong. Like a team where each
          member specializes in fixing others' mistakes.

        - Ridge: Finds the best linear combination of features. Simple but
          can't capture complex patterns like "scoring drops when tired AND
          facing elite defense".
        """
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=100,       # Number of trees
                max_depth=6,            # How deep each tree can go
                learning_rate=0.1,      # How much each tree contributes
                random_state=42,        # For reproducibility
                n_jobs=-1               # Use all CPU cores
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,    # Minimum samples to split a node
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:  # ridge (default fallback)
            return Ridge(alpha=1.0)     # alpha controls regularization strength

    def train(
        self,
        df: pd.DataFrame,
        target: str = "PTS",
        feature_columns: list[str] = None,
        test_size: float = 0.2,
        weight_column: str = None,
    ) -> dict:
        """
        Train the model on historical data.

        HOW TRAINING WORKS:
        1. Select a model on the earliest dates with expanding validation folds.
        2. Estimate residual uncertainty on a separate calibration period.
        3. Evaluate once on the final untouched dates.
        4. Refit the selected configuration on all data for production.

        Args:
            df: DataFrame with engineered features
            target: What to predict ("PTS", "AST", or "REB")
            feature_columns: Which columns to use as features
                            (if None, auto-detect numeric columns)
            test_size: Fraction of data to use for testing (default 0.2 = 20%)

        Returns:
            Dictionary with training metrics:
            {
                "mae": 3.5,      # Mean Absolute Error
                "r2": 0.72,      # R-squared score
                "train_samples": 2939,
                "test_samples": 735
            }
        """
        if weight_column is not None:
            raise ValueError("Outcome-based sample weights are not permitted; use pregame eligibility")
        from models.validation import fit_validated
        return fit_validated(self, df, target, feature_columns, test_size=test_size)

    def predict(
        self,
        features: pd.DataFrame | dict | np.ndarray
    ) -> float | np.ndarray:
        """
        Make predictions for new games.

        Args:
            features: Can be:
                - DataFrame with feature columns
                - Dict with feature values
                - Numpy array with feature values

        Returns:
            Predicted value(s) for the target stat

        Example:
            >>> # Predict LeBron's points for next game
            >>> features = {
            ...     "rolling_avg_pts_5": 25.5,
            ...     "rolling_avg_pts_10": 26.0,
            ...     "is_home": 1,
            ...     "days_rest": 2,
            ...     ...
            ... }
            >>> predicted_pts = predictor.predict(features)
            >>> print(f"Predicted: {predicted_pts:.1f} points")
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")

        # Convert input to array
        if isinstance(features, dict):
            # Single prediction from dict
            missing = [col for col in self.feature_columns if col not in features]
            if missing:
                raise ValueError(f"Missing pregame features: {missing}")
            X = np.array([[features[col] for col in self.feature_columns]])
        elif isinstance(features, pd.DataFrame):
            X = features[self.feature_columns].values
        else:
            X = features

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = np.maximum(0, self.model.predict(X_scaled))

        # Return single value if single prediction, else array
        if len(predictions) == 1:
            return round(float(predictions[0]), 1)
        return predictions

    def predict_player_game(
        self,
        player_name: str,
        features_df: pd.DataFrame,
        n_recent_games: int = 10,
        is_home: bool = None,
        game_date=None,
    ) -> dict:
        """
        Predict a player's stats based on their recent performance.

        This is a convenience method that:
        1. Finds the player's most recent games
        2. Uses their most recent feature values
        3. Makes a prediction

        Args:
            player_name: Full name like "LeBron James"
            features_df: DataFrame with all player features
            n_recent_games: How many recent games to consider for features

        Returns:
            Dictionary with prediction and context:
            {
                "player": "LeBron James",
                "predicted_pts": 26.5,
                "recent_avg": 25.3,
                "season_avg": 25.0,
                "confidence": "medium"
            }
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")

        # Filter to player
        player_df = features_df[
            features_df["PLAYER_NAME"].str.lower() == player_name.lower()
        ]

        if player_df.empty:
            return {"error": f"Player '{player_name}' not found"}

        # Get most recent game's features
        from utils.pregame_features import next_game_features
        player_df = player_df.assign(_sort_date=pd.to_datetime(player_df["GAME_DATE"], format="mixed"))
        player_df = player_df.sort_values("_sort_date", ascending=False)
        recent_features = next_game_features(player_df, game_date, is_home)

        # Make prediction
        prediction = self.predict({
            col: recent_features[col]
            for col in self.feature_columns
            if col in recent_features
        })

        # Get context stats
        recent_avg = player_df.head(n_recent_games)[self.target].mean()
        season_avg = player_df[self.target].mean()

        # Low variance alone cannot establish confidence in a forecast.
        confidence = "low"
        residuals = getattr(self, "calibration_residuals", None)
        interval = None
        if residuals is not None and len(residuals):
            interval = [round(max(0, prediction + float(q)), 1)
                        for q in np.quantile(residuals, [0.1, 0.9])]

        return {
            "player": player_name,
            f"predicted_{self.target.lower()}": prediction,
            "recent_avg": round(recent_avg, 1),
            "season_avg": round(season_avg, 1),
            "confidence": confidence,
            "prediction_interval_80": interval,
            "games_analyzed": len(player_df)
        }

    def _get_default_features(self, df: pd.DataFrame) -> list[str]:
        """
        Auto-detect which columns to use as features.

        IMPORTANT - DATA LEAKAGE PREVENTION:
        We must ONLY use features that are known BEFORE the game happens!

        We EXCLUDE:
        - Same-game stats (FGM, FGA, etc.) - these happen DURING the game
        - Targets (PTS, AST, REB) - what we're predicting
        - Identifiers (names, dates)

        We INCLUDE:
        - Rolling averages (from previous games)
        - Home/away, rest days (known before game)
        - Opponent defensive ratings (known before game)
        """
        from models.validation import safe_features
        return safe_features(df)

    def _print_feature_importance(self, top_n: int = 10):
        """
        Print the most important features for prediction.

        WHY FEATURE IMPORTANCE MATTERS:
        Understanding which features the model relies on helps us:
        1. Trust the model (does it make sense?)
        2. Improve features (add more like the important ones)
        3. Debug issues (is it using the right signals?)
        """
        if not hasattr(self.model, "feature_importances_"):
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        print(f"\nTop {top_n} Most Important Features:")
        for i, idx in enumerate(indices, 1):
            feat_name = self.feature_columns[idx]
            importance = importances[idx]
            bar = "█" * int(importance * 50)
            print(f"  {i}. {feat_name}: {importance:.3f} {bar}")

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Return feature importance as a DataFrame for analysis.

        Returns:
            DataFrame with columns: feature, importance
            Sorted by importance descending
        """
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        return importance_df

    def prune_weak_features(self, threshold: float = 0.01) -> list[str]:
        """
        Get list of features with importance above threshold.

        WHY PRUNE:
        Removing weak features can reduce overfitting and speed up predictions.
        Features with <1% importance often add noise rather than signal.

        Args:
            threshold: Minimum importance to keep (default 0.01 = 1%)

        Returns:
            List of feature names that are above the threshold
        """
        importance_df = self.get_feature_importance_df()

        if importance_df.empty:
            return self.feature_columns

        strong_features = importance_df[
            importance_df["importance"] >= threshold
        ]["feature"].tolist()

        print(f"Pruned from {len(self.feature_columns)} to {len(strong_features)} features")
        print(f"Removed: {set(self.feature_columns) - set(strong_features)}")

        return strong_features

    def tune_hyperparameters(
        self,
        df: pd.DataFrame,
        target: str = None,
        feature_columns: list[str] = None,
        cv: int = 5
    ) -> dict:
        """
        Select a configuration without looking at calibration or final test outcomes.

        WHY TUNE:
        Compare regularized models and rolling baselines on chronological folds.
        Improvement is measured, never assumed.

        Args:
            df: DataFrame with engineered features
            target: Which stat to predict (uses self.target if not provided)
            feature_columns: Which features to use
            cv: Number of cross-validation folds

        Returns:
            Dictionary with best parameters found
        """
        from models.validation import fit_validated
        fit_validated(self, df, target or self.target or "PTS", feature_columns, tune=True, cv=cv)
        return {"selected_candidate": self.metrics["selected_candidate"]}

    def predict_batch(
        self,
        players: list[str],
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions for multiple players at once.

        Useful for "Best Props" feature where we need to predict
        all players playing today and rank them.

        Args:
            players: List of player names
            features_df: DataFrame with all player features

        Returns:
            DataFrame with predictions and confidence for each player
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")

        results = []

        for player in players:
            try:
                pred = self.predict_player_game(player, features_df)
                if "error" not in pred:
                    results.append({
                        "player": player,
                        f"pred_{self.target.lower()}": pred[f"predicted_{self.target.lower()}"],
                        "recent_avg": pred["recent_avg"],
                        "season_avg": pred["season_avg"],
                        "confidence": pred["confidence"]
                    })
            except Exception:
                continue

        return pd.DataFrame(results)

    def save(self, filepath: str = None):
        """
        Save the trained model to disk.

        WHY SAVE MODELS:
        Training can take time. Once trained, we save the model so we
        can load it instantly later without retraining.
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")

        if filepath is None:
            from utils.league_config import get_config
            models_dir = get_config("nba").models_dir
            models_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(models_dir / f"{self.target.lower()}_predictor.pkl")

        # Save everything needed to make predictions
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "target": self.target,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "calibration_residuals": getattr(self, "calibration_residuals", None)
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "StatPredictor":
        """
        Load a trained model from disk.

        Example:
            >>> predictor = StatPredictor.load("models/pts_predictor.pkl")
            >>> prediction = predictor.predict(features)
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        predictor = cls(model_type=model_data["model_type"])
        predictor.model = model_data["model"]
        predictor.scaler = model_data["scaler"]
        predictor.feature_columns = model_data["feature_columns"]
        predictor.target = model_data["target"]
        predictor.metrics = model_data["metrics"]
        predictor.calibration_residuals = model_data.get("calibration_residuals")
        predictor.is_trained = True

        print(f"Loaded {predictor.target} predictor (MAE: {predictor.metrics['mae']})")

        return predictor


# =============================================================================
# MULTI-STAT PREDICTOR
# =============================================================================

class MultiStatPredictor:
    """
    Convenience class for predicting multiple stats at once.

    Instead of training separate models manually, this class handles
    all of them together.

    Example:
        >>> predictor = MultiStatPredictor()
        >>> predictor.train_all(features_df)
        >>> predictions = predictor.predict_all(player_features)
        >>> print(predictions)
        {"PTS": 26.5, "AST": 7.2, "REB": 8.1}
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.predictors = {}
        self.targets = ["PTS", "AST", "REB"]

    def train_all(self, df: pd.DataFrame) -> dict:
        """Train models for PTS, AST, and REB."""
        results = {}

        for target in self.targets:
            print(f"\n{'='*60}")
            predictor = StatPredictor(model_type=self.model_type)
            metrics = predictor.train(df, target=target)
            self.predictors[target] = predictor
            results[target] = metrics

        return results

    def predict_all(self, features: pd.DataFrame | dict) -> dict:
        """Predict all stats at once."""
        predictions = {}

        for target, predictor in self.predictors.items():
            predictions[target] = predictor.predict(features)

        return predictions

    def save_all(self, directory: str = None):
        """Save all models."""
        if directory is None:
            from utils.league_config import get_config
            directory = str(get_config("nba").models_dir)
            os.makedirs(directory, exist_ok=True)

        for target, predictor in self.predictors.items():
            filepath = os.path.join(directory, f"{target.lower()}_predictor.pkl")
            predictor.save(filepath)

    def load_all(self, directory: str = None):
        """Load all models."""
        if directory is None:
            from utils.league_config import get_config
            directory = str(get_config("nba").models_dir)

        for target in self.targets:
            filepath = os.path.join(directory, f"{target.lower()}_predictor.pkl")
            if os.path.exists(filepath):
                self.predictors[target] = StatPredictor.load(filepath)
