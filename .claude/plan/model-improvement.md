# Implementation Plan: Model Improvement — XGBoost + Elo + Feature Upgrades

## Task Type
- [x] Backend (ML model + prediction pipeline)

## Problem Analysis

### Current Architecture (Heuristic Formula)
```
pred_home = (rolling_ppg × 0.65 + opp_rolling_ppg × 0.35)
          + form_momentum_adjustment
          + margin_adjustment
          + h2h_boost
          + home_court_adv (+2.5)
          × pace_factor
          - injury_penalty
          - calibration_bias
```

**Problems:**
1. Weights (0.65/0.35) are hand-tuned, not learned from data
2. No rest/fatigue feature (B2B = -2.5 PPG on average)
3. `feature_engineering.py` is built but NEVER USED
4. Normal distribution for props hit probability (wrong for Poisson stats like AST, 3PM)
5. Calibration only corrects mean bias, not conditional patterns

### What We Have Available
- `utils/feature_engineering.py`: Full feature engineering pipeline (rolling avgs, rest days, opponent defense, pace, position, matchup difficulty)
- `data/player_game_logs.csv`: Full season historical data (~100k rows)
- `data/team_defensive_stats.csv`: Team defensive ratings, pace
- `utils/prediction_tracker.py`: Graded prediction history for calibration

---

## Technical Solution

### Priority Stack (3 tiers of improvement)

**Tier 1 — Quick wins (~2 hours, big impact)**
- Add rest/back-to-back fatigue to game predictor
- Use Vegas spread as prior anchor (blend model 50/50 with market)
- Fix props probability: use Poisson distribution for low-count stats (AST, 3PM, BLK)

**Tier 2 — Real ML model (~4 hours, largest accuracy gain)**
- Train XGBoost regressor for team game scores
- Features: rolling PPG, opp rolling PPG, rest days, H2H win rate, home, pace, B2B, form W/L pct
- Train/store model as `data/model_game_xgb.pkl`
- Fallback to heuristic if XGB model not found

**Tier 3 — Advanced (optional, future)**
- Per-player XGBoost for props (one model per stat type)
- Elo rating system for team strength
- Market-calibrated confidence intervals

---

## Implementation Steps

### Step 1 — Add Rest/Fatigue to Game Predictor
**File:** `utils/game_predictor.py`
**Impact:** High — B2B has measurable -2.5 to -4 PPG effect

```python
# In predict_game(), compute rest days per team from self._scores
def _get_rest_days(self, team: str) -> int:
    """Days since last game. 0 = back-to-back, 1 = normal, 2+ = rested."""
    if self._scores.empty:
        return 1
    team_games = self._scores[self._scores["team"] == team].sort_values("_date", ascending=False)
    if len(team_games) < 2:
        return 1
    delta = (team_games.iloc[0]["_date"] - team_games.iloc[1]["_date"]).days
    return min(delta, 7)

# In predict_game() after pace adjustment:
B2B_PENALTY = 2.5   # pts deducted for back-to-back
REST_BONUS  = 0.8   # small bonus for 3+ days rest

home_rest = self._get_rest_days(home)
away_rest = self._get_rest_days(away)
if home_rest == 0:   pred_home -= B2B_PENALTY
if away_rest == 0:   pred_away -= B2B_PENALTY
if home_rest >= 3:   pred_home += REST_BONUS
if away_rest >= 3:   pred_away += REST_BONUS
```

### Step 2 — Vegas Line Anchor Blend
**File:** `utils/game_predictor.py` → `get_pick()`
**Impact:** High — market line is very calibrated, blending reduces model error

```python
# When actual_spread is known, blend model spread with market:
MARKET_BLEND = 0.35   # trust market 35%, our model 65%

def get_pick(self, prediction, home, away, actual_spread=None, actual_total=None):
    model_spread = prediction["predicted_spread"]
    model_total  = prediction["predicted_total"]

    # Blend with market if available
    if actual_spread is not None:
        blended_spread = model_spread * (1 - MARKET_BLEND) + (-actual_spread) * MARKET_BLEND
    else:
        blended_spread = model_spread

    if actual_total is not None:
        blended_total = model_total * (1 - MARKET_BLEND) + actual_total * MARKET_BLEND
    else:
        blended_total = model_total

    # Use blended values for edge calculation
    edge_spread = blended_spread + actual_spread   # home covers?
    ...
```

### Step 3 — XGBoost Game Score Model (Tier 2)

#### 3a. Training Script
**New file:** `scripts/train_game_model.py`

```python
"""
Train XGBoost model for game score prediction.
Run once: python3 scripts/train_game_model.py
Creates: data/model_home_xgb.pkl, data/model_away_xgb.pkl
"""
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from utils.kaggle_loader import load_player_game_logs, load_team_defensive_stats

def build_team_game_features(player_logs_df, team_def_df):
    """
    Aggregate player logs to team-game level and engineer features.
    Returns DataFrame with one row per team-game.
    """
    df = player_logs_df.copy()
    df["_date"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")

    # Extract team/opponent from MATCHUP
    df["team"] = df["MATCHUP"].str.extract(r'^([A-Z]{2,3})\s+(?:vs\.|@)')[0]
    df["opponent"] = df["MATCHUP"].str.extract(r'(?:vs\.|@)\s+([A-Z]{2,3})')[0]
    df["is_home"] = df["MATCHUP"].str.contains(r"\bvs\.", regex=True, na=False).astype(int)

    # Team game totals
    team_games = (
        df.groupby(["team", "_date", "opponent", "is_home", "WL"])
        .agg(team_pts=("PTS", "sum"))
        .reset_index()
    )
    team_games = team_games.sort_values(["team", "_date"])

    # Rolling features per team (using shift to avoid leakage)
    team_games["roll_ppg_10"] = (
        team_games.groupby("team")["team_pts"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
    team_games["roll_ppg_5"] = (
        team_games.groupby("team")["team_pts"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    )
    team_games["roll_wl_10"] = (
        team_games.groupby("team")["WL"]
        .transform(lambda x: (x.shift(1).rolling(10, min_periods=3).apply(
            lambda w: (w == "W").sum() / len(w), raw=False
        )))
    )

    # Rest days
    team_games["rest_days"] = (
        team_games.groupby("team")["_date"]
        .diff().dt.days.fillna(3).clip(0, 10)
    )
    team_games["is_b2b"] = (team_games["rest_days"] <= 1).astype(int)

    # Merge opponent rolling PPG as defensive proxy
    opp_stats = team_games[["team", "_date", "roll_ppg_10", "roll_ppg_5"]].copy()
    opp_stats.columns = ["opponent", "_date", "opp_roll_ppg_10", "opp_roll_ppg_5"]

    team_games = team_games.merge(
        opp_stats, on=["opponent", "_date"], how="left"
    )

    # Target: actual team pts
    # Features: roll_ppg_10, roll_ppg_5, roll_wl_10, is_home, rest_days, is_b2b,
    #           opp_roll_ppg_10, opp_roll_ppg_5
    return team_games.dropna(subset=["roll_ppg_10", "opp_roll_ppg_10", "team_pts"])


def train():
    print("Loading data...")
    logs = load_player_game_logs(num_seasons=3)  # 3 seasons of data
    team_def = load_team_defensive_stats(num_seasons=3)

    print("Building features...")
    team_games = build_team_game_features(logs, team_def)

    features = ["roll_ppg_10", "roll_ppg_5", "roll_wl_10",
                 "is_home", "rest_days", "is_b2b",
                 "opp_roll_ppg_10", "opp_roll_ppg_5"]
    target = "team_pts"

    # Time-series aware split (no future leakage)
    team_games = team_games.sort_values("_date")
    split_idx = int(len(team_games) * 0.85)  # 85% train
    train_df = team_games.iloc[:split_idx]
    test_df  = team_games.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test  = test_df[features]
    y_test  = test_df[target]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Test MAE: {mae:.2f} pts per team per game")

    # Save
    with open("data/model_game_xgb.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved to data/model_game_xgb.pkl")

    # Baseline comparison
    baseline_mae = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
    print(f"Baseline (mean) MAE: {baseline_mae:.2f}")
    print(f"XGB improvement: {(1 - mae/baseline_mae)*100:.1f}%")


if __name__ == "__main__":
    train()
```

#### 3b. Wire XGB into GamePredictor
**File:** `utils/game_predictor.py`

```python
# At class init, try to load XGB model
import pickle
from pathlib import Path

class GamePredictor:
    _XGB_PATH = Path(__file__).parent.parent / "data" / "model_game_xgb.pkl"

    def __init__(self, team_def_df, player_logs_df):
        ...
        self._xgb = self._load_xgb()

    def _load_xgb(self):
        try:
            with open(self._XGB_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _predict_xgb(self, team: str, opp: str, is_home: int) -> float | None:
        """XGB prediction for team score. Returns None if model not available."""
        if self._xgb is None:
            return None
        form = self._get_form(team)
        opp_form = self._get_form(opp)
        rest = self._get_rest_days(team)

        features = {
            "roll_ppg_10": form.get("rolling_ppg") or 112.0,
            "roll_ppg_5":  form.get("rolling_ppg") or 112.0,
            "roll_wl_10":  form.get("wins", 5) / max(form.get("wins", 5) + form.get("losses", 5), 1),
            "is_home":     is_home,
            "rest_days":   rest,
            "is_b2b":      1 if rest == 0 else 0,
            "opp_roll_ppg_10": opp_form.get("rolling_ppg") or 112.0,
            "opp_roll_ppg_5":  opp_form.get("rolling_ppg") or 112.0,
        }
        import pandas as pd
        X = pd.DataFrame([features])
        return float(self._xgb.predict(X)[0])

    def predict_game(self, home, away, home_injuries=None, away_injuries=None):
        ...
        # After existing formula computation:
        xgb_home = self._predict_xgb(home, away, is_home=1)
        xgb_away = self._predict_xgb(away, home, is_home=0)

        XGB_WEIGHT = 0.45   # blend: 55% heuristic, 45% XGB
        if xgb_home is not None and xgb_away is not None:
            pred_home = pred_home * (1 - XGB_WEIGHT) + xgb_home * XGB_WEIGHT
            pred_away = pred_away * (1 - XGB_WEIGHT) + xgb_away * XGB_WEIGHT
        ...
```

### Step 4 — Fix Props Probability Distribution
**File:** `utils/props_cache.py` and `utils/prop_calculator.py`
**Impact:** Medium — better hit rate estimates for low-count stats (AST ≤ 5, 3PM, BLK)

```python
# prop_calculator.py — add Poisson model for low-count stats
from scipy.stats import poisson, norm

def calculate_hit_probability_smart(prediction, line, std_dev, stat_type, direction="over"):
    """
    Use Poisson for discrete low-count stats (AST, 3PM, BLK, STL).
    Use Normal for high-count continuous stats (PTS, REB, combos).
    """
    LOW_COUNT_STATS = {"AST", "FG3M", "BLK", "STL"}

    if stat_type in LOW_COUNT_STATS and prediction < 10:
        # Poisson is more accurate for discrete count data
        lam = max(prediction, 0.1)
        if direction == "over":
            return 1 - poisson.cdf(int(line), mu=lam)
        else:
            return poisson.cdf(int(line) - 1, mu=lam)
    else:
        # Normal for PTS, REB, combo stats
        if std_dev == 0 or std_dev is None:
            std_dev = max(prediction * 0.2, 3)
        if direction == "over":
            return 1 - norm.cdf(line, loc=prediction, scale=std_dev)
        else:
            return norm.cdf(line, loc=prediction, scale=std_dev)
```

### Step 5 — Better Calibration (Conditional Patterns)
**File:** `utils/prediction_tracker.py`
**Impact:** Medium — catch systematic errors (e.g., always under in fast-paced games)

```python
# Extended calibration: track errors by pace tier and home/away
# Current: just home_score_bias and away_score_bias
# New: add pace_hi_bias (error when both teams pace > 100), pace_lo_bias, etc.

calibration_extended = {
    "home_score_bias": mean error for home team scores,
    "away_score_bias": mean error for away team scores,
    "spread_bias":     mean error for spread (model - actual),
    "total_bias":      mean error for total (model - actual),
    "fast_game_bias":  error when pace > 100 (model tends to under/over predict),
    "blowout_bias":    error when actual margin > 15 pts,
}
```

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `scripts/train_game_model.py` | **Create** | XGBoost training script (run once) |
| `utils/game_predictor.py` | Modify | Add rest days, XGB blend, rest bonus |
| `utils/prop_calculator.py` | Modify | Poisson distribution for low-count stats |
| `utils/prediction_tracker.py` | Modify | Extended conditional calibration |
| `data/model_game_xgb.pkl` | **Create** | Trained model artifact (generated by script) |

---

## Expected Accuracy Gains

| Improvement | Expected MAE Reduction | Notes |
|-------------|----------------------|-------|
| Rest/B2B factor | ~0.5 pts/game | Well-documented NBA effect |
| Vegas line anchor | ~1.0 pts/game spread | Market is very accurate |
| XGBoost model | ~1.5-2.5 pts/game | vs current formula |
| Poisson props | Higher hit rate accuracy | Especially AST, 3PM |

Typical NBA game predictor MAE (Mean Absolute Error):
- Random / league average: ~12 pts
- Current formula: ~9-10 pts estimated
- With XGB + improvements: ~7-8 pts target

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| XGB needs data for training | Fallback to heuristic if `model_game_xgb.pkl` missing |
| Overfitting on training data | TimeSeriesSplit validation, simple feature set |
| Poisson breaks for combos | Only apply to single low-count stats |
| Vegas line not always available | Only blend when odds present, else pure model |
| Rest days not in current data | Use `_scores` table to compute from last game date |

---

## Installation Requirement

```bash
pip install xgboost scikit-learn
```

(Already in most ML environments, add to requirements.txt)

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (external models unavailable)
- GEMINI_SESSION: N/A (external models unavailable)
