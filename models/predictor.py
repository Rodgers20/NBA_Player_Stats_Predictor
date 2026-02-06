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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
    print("Note: XGBoost not available, using Random Forest instead")


class NBAPredictor:
    """
    Main class for NBA player stats prediction.

    This class handles:
    1. Training models for PTS, AST, REB predictions
    2. Making predictions for new games
    3. Saving/loading trained models
    4. Evaluating model performance

    Example Usage:
    --------------
    >>> from models.predictor import NBAPredictor
    >>> from utils.feature_engineering import engineer_features
    >>>
    >>> # Load and prepare data
    >>> df = pd.read_csv("data/player_game_logs.csv")
    >>> features_df = engineer_features(df)
    >>>
    >>> # Train the model
    >>> predictor = NBAPredictor()
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
        test_size: float = 0.2
    ) -> dict:
        """
        Train the model on historical data.

        HOW TRAINING WORKS:
        1. Split data into training (80%) and testing (20%) sets
        2. Fit (train) the model on training data
        3. Evaluate on test data to see real-world performance
        4. Store the model for later predictions

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
        print(f"\n{'='*50}")
        print(f"TRAINING MODEL: Predicting {target}")
        print(f"Model type: {self.model_type}")
        print(f"{'='*50}")

        # Store target
        self.target = target

        # Auto-detect features if not provided
        if feature_columns is None:
            feature_columns = self._get_default_features(df)

        self.feature_columns = feature_columns
        print(f"\nUsing {len(feature_columns)} features")

        # Prepare data
        # Remove rows with missing values in features or target
        required_cols = feature_columns + [target]
        df_clean = df.dropna(subset=required_cols)

        X = df_clean[feature_columns].values
        y = df_clean[target].values

        print(f"Total samples: {len(X)}")

        # Split into train/test
        # IMPORTANT: We use random_state for reproducibility
        # This means you'll get the same split every time
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Scale features (important for some models)
        # Scaling makes all features have similar ranges (mean=0, std=1)
        # This helps models that are sensitive to feature scales
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train the model
        print(f"\nTraining {self.model_type} model...")
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.metrics = {
            "mae": round(mae, 2),
            "r2": round(r2, 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        self.is_trained = True

        # Print results
        print(f"\n{'='*50}")
        print("TRAINING COMPLETE!")
        print(f"{'='*50}")
        print(f"Mean Absolute Error: {mae:.2f} {target.lower()}")
        print(f"  → Predictions are off by ~{mae:.1f} {target.lower()} on average")
        print(f"R² Score: {r2:.3f}")
        print(f"  → Model explains {r2*100:.1f}% of variance in {target.lower()}")

        # Show feature importance (if available)
        self._print_feature_importance()

        return self.metrics

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
            X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        elif isinstance(features, pd.DataFrame):
            X = features[self.feature_columns].values
        else:
            X = features

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        # Return single value if single prediction, else array
        if len(predictions) == 1:
            return round(float(predictions[0]), 1)
        return predictions

    def predict_player_game(
        self,
        player_name: str,
        features_df: pd.DataFrame,
        n_recent_games: int = 10
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
        player_df = player_df.sort_values("GAME_DATE", ascending=False)
        recent_features = player_df.iloc[0]

        # Make prediction
        prediction = self.predict({
            col: recent_features[col]
            for col in self.feature_columns
            if col in recent_features
        })

        # Get context stats
        recent_avg = player_df.head(n_recent_games)[self.target].mean()
        season_avg = player_df[self.target].mean()

        # Determine confidence based on consistency
        recent_std = player_df.head(n_recent_games)[self.target].std()
        if recent_std < 5:
            confidence = "high"
        elif recent_std < 8:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "player": player_name,
            f"predicted_{self.target.lower()}": prediction,
            "recent_avg": round(recent_avg, 1),
            "season_avg": round(season_avg, 1),
            "confidence": confidence,
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
        exclude = [
            # Identifiers
            "PLAYER_NAME", "GAME_DATE", "MATCHUP", "SEASON",
            "Player_ID", "Game_ID", "SEASON_ID", "VIDEO_AVAILABLE",

            # Targets (what we predict - don't use these!)
            "PTS", "AST", "REB", "STL", "BLK", "TOV",

            # SAME-GAME STATS (data leakage - these happen DURING the game!)
            # We can't know these before the game, so we can't use them
            "MIN",      # Minutes played in THIS game
            "FGM", "FGA", "FG_PCT",    # Field goals in THIS game
            "FG3M", "FG3A", "FG3_PCT", # 3-pointers in THIS game
            "FTM", "FTA", "FT_PCT",    # Free throws in THIS game
            "OREB", "DREB",            # Rebounds in THIS game
            "PLUS_MINUS",              # Plus/minus in THIS game

            # Other
            "WL", "opponent"
        ]

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter out excluded columns
        features = [col for col in numeric_cols if col not in exclude]

        return features

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
        Use GridSearchCV to find optimal hyperparameters.

        WHY TUNE:
        Default hyperparameters work okay, but tuning can improve accuracy
        by 5-15%. Takes longer but worth it for production models.

        Args:
            df: DataFrame with engineered features
            target: Which stat to predict (uses self.target if not provided)
            feature_columns: Which features to use
            cv: Number of cross-validation folds

        Returns:
            Dictionary with best parameters found
        """
        if target is None:
            target = self.target or "PTS"

        if feature_columns is None:
            feature_columns = self._get_default_features(df)

        print(f"\n{'='*50}")
        print(f"HYPERPARAMETER TUNING FOR {target}")
        print(f"{'='*50}")

        # Prepare data
        df_clean = df.dropna(subset=feature_columns + [target])
        X = df_clean[feature_columns].values
        y = df_clean[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Define parameter grids based on model type
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.15],
                "subsample": [0.8, 1.0],
            }
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        else:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [6, 10, 15],
                "min_samples_split": [2, 5, 10],
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        print(f"Testing {sum(len(v) for v in param_grid.values())} parameter combinations...")
        print("This may take several minutes...")

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_scaled, y_train)

        print(f"\n{'='*50}")
        print("TUNING COMPLETE!")
        print(f"{'='*50}")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: {-grid_search.best_score_:.2f}")

        # Update model with best estimator
        self.model = grid_search.best_estimator_
        self.feature_columns = feature_columns
        self.target = target

        # Evaluate on test set
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        print(f"Test MAE: {test_mae:.2f}")
        print(f"Test R²: {test_r2:.3f}")

        self.is_trained = True
        self.metrics = {
            "mae": round(test_mae, 2),
            "r2": round(test_r2, 3),
            "best_params": grid_search.best_params_
        }

        return grid_search.best_params_

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
            # Default path in models directory
            models_dir = os.path.dirname(__file__)
            filepath = os.path.join(models_dir, f"{self.target.lower()}_predictor.pkl")

        # Save everything needed to make predictions
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "target": self.target,
            "model_type": self.model_type,
            "metrics": self.metrics
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "NBAPredictor":
        """
        Load a trained model from disk.

        Example:
            >>> predictor = NBAPredictor.load("models/pts_predictor.pkl")
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
            predictor = NBAPredictor(model_type=self.model_type)
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
            directory = os.path.dirname(__file__)

        for target, predictor in self.predictors.items():
            filepath = os.path.join(directory, f"{target.lower()}_predictor.pkl")
            predictor.save(filepath)

    def load_all(self, directory: str = None):
        """Load all models."""
        if directory is None:
            directory = os.path.dirname(__file__)

        for target in self.targets:
            filepath = os.path.join(directory, f"{target.lower()}_predictor.pkl")
            if os.path.exists(filepath):
                self.predictors[target] = NBAPredictor.load(filepath)
