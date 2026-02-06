# utils/prop_calculator.py
"""
Prop Bet Probability Calculator
===============================
Calculate hit probabilities for player props based on model predictions
and historical hit rates.

HOW IT WORKS:
1. Get model prediction (e.g., LeBron predicted 26.5 PTS)
2. Compare to common prop lines (e.g., Over 24.5, Over 26.5)
3. Calculate probability of hitting each line
4. Find the best line with highest edge

PROBABILITY CALCULATION:
Uses normal distribution assumption:
- Mean = model prediction
- Std = player's recent variance
- P(Over line) = 1 - CDF(line)

This is a simplification but works well for betting analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from scipy import stats


# =============================================================================
# COMMON PROP LINES
# =============================================================================

# Standard lines offered by sportsbooks for each stat type
COMMON_LINES = {
    "PTS": [9.5, 14.5, 19.5, 24.5, 29.5, 34.5, 39.5],
    "AST": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
    "REB": [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
    "FG3M": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    "STL": [0.5, 1.5, 2.5],
    "BLK": [0.5, 1.5, 2.5],
    "PTS+REB": [14.5, 19.5, 24.5, 29.5, 34.5, 39.5],
    "PTS+AST": [14.5, 19.5, 24.5, 29.5, 34.5, 39.5],
    "PTS+REB+AST": [19.5, 24.5, 29.5, 34.5, 39.5, 44.5, 49.5],
}


# =============================================================================
# PROBABILITY CALCULATIONS
# =============================================================================

def calculate_hit_probability(
    prediction: float,
    line: float,
    std_dev: float,
    direction: str = "over"
) -> float:
    """
    Calculate probability of hitting a prop line.

    Uses normal distribution:
    P(X > line) = 1 - CDF(line)

    Args:
        prediction: Model's predicted value
        line: The prop line (e.g., 24.5 points)
        std_dev: Standard deviation from historical data
        direction: "over" or "under"

    Returns:
        Probability between 0 and 1

    Example:
        >>> prob = calculate_hit_probability(26.5, 24.5, 5.0, "over")
        >>> print(f"{prob:.1%}")
        65.5%
    """
    # Prevent division by zero
    if std_dev == 0 or std_dev is None:
        std_dev = max(prediction * 0.2, 3)  # Default to 20% of prediction or min 3

    if direction == "over":
        # Probability that actual value exceeds the line
        prob = 1 - stats.norm.cdf(line, loc=prediction, scale=std_dev)
    else:
        # Probability that actual value is below the line
        prob = stats.norm.cdf(line, loc=prediction, scale=std_dev)

    return round(prob, 4)


def calculate_edge(
    hit_probability: float,
    implied_odds: float = 0.524  # -110 odds = 52.4% implied
) -> float:
    """
    Calculate the edge over the sportsbook.

    Edge = (True Prob - Implied Prob) / Implied Prob

    Positive edge = profitable bet over time
    Negative edge = sportsbook has advantage

    Args:
        hit_probability: Our calculated probability
        implied_odds: Sportsbook's implied probability (default -110 = 52.4%)

    Returns:
        Edge as a decimal (0.10 = 10% edge)
    """
    if implied_odds == 0:
        return 0

    edge = (hit_probability - implied_odds) / implied_odds
    return round(edge, 4)


def find_best_line(
    prediction: float,
    recent_stats: pd.Series,
    stat_type: str = "PTS",
    min_prob: float = 0.55,
    max_prob: float = 0.75
) -> Tuple[Optional[float], float, str]:
    """
    Find the best prop line for a player's prediction.

    STRATEGY:
    - We want lines where our probability is between 55-75%
    - Too high (>75%): Line is obvious, probably low value
    - Too low (<55%): Not enough edge to bet

    Args:
        prediction: Model's predicted value
        recent_stats: Series of recent game values for std calculation
        stat_type: Which stat type (PTS, AST, REB, etc.)
        min_prob: Minimum probability to consider
        max_prob: Maximum probability to consider

    Returns:
        Tuple of (best_line, hit_probability, direction)
        Returns (None, 0, "") if no good line found
    """
    lines = COMMON_LINES.get(stat_type, [prediction - 2.5, prediction + 2.5])

    # Calculate standard deviation from recent games
    if len(recent_stats) > 0:
        std_dev = recent_stats.std()
        if pd.isna(std_dev) or std_dev == 0:
            std_dev = prediction * 0.2
    else:
        std_dev = prediction * 0.2

    best_line = None
    best_prob = 0
    best_direction = ""
    best_edge = -1

    for line in lines:
        # Check over
        over_prob = calculate_hit_probability(prediction, line, std_dev, "over")
        if min_prob <= over_prob <= max_prob:
            edge = calculate_edge(over_prob)
            if edge > best_edge:
                best_edge = edge
                best_prob = over_prob
                best_line = line
                best_direction = "over"

        # Check under
        under_prob = calculate_hit_probability(prediction, line, std_dev, "under")
        if min_prob <= under_prob <= max_prob:
            edge = calculate_edge(under_prob)
            if edge > best_edge:
                best_edge = edge
                best_prob = under_prob
                best_line = line
                best_direction = "under"

    return best_line, best_prob, best_direction


# =============================================================================
# HIT RATE ANALYSIS
# =============================================================================

def calculate_historical_hit_rate(
    games: pd.Series,
    line: float,
    direction: str = "over"
) -> float:
    """
    Calculate historical hit rate for a given line.

    Args:
        games: Series of stat values from past games
        line: The prop line to check
        direction: "over" or "under"

    Returns:
        Hit rate as a decimal (0.70 = 70%)
    """
    if len(games) == 0:
        return 0.0

    if direction == "over":
        hits = (games >= line).sum()
    else:
        hits = (games < line).sum()

    return round(hits / len(games), 3)


def get_streak_info(
    games: pd.Series,
    line: float,
    direction: str = "over"
) -> dict:
    """
    Get streak information for a prop line.

    Returns:
        Dictionary with current streak and max streak info
    """
    if len(games) == 0:
        return {"current_streak": 0, "max_streak": 0}

    if direction == "over":
        hits = (games >= line).values
    else:
        hits = (games < line).values

    # Current streak (from most recent)
    current_streak = 0
    for hit in hits:
        if hit:
            current_streak += 1
        else:
            break

    # Max streak
    max_streak = 0
    streak = 0
    for hit in hits:
        if hit:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "current_streak": current_streak,
        "max_streak": max_streak
    }


# =============================================================================
# BEST PROPS GENERATION
# =============================================================================

def generate_best_props(
    predictions_df: pd.DataFrame,
    game_logs_df: pd.DataFrame,
    stat_type: str = "PTS",
    top_n: int = 20,
    min_games: int = 5
) -> pd.DataFrame:
    """
    Generate ranked list of best prop bets for today.

    This is the MAIN FUNCTION for the Best Props page.
    It analyzes all players and returns the top picks.

    Args:
        predictions_df: DataFrame with player predictions
                       Must have columns: player, pred_{stat_type}
        game_logs_df: Historical game logs for std dev calculation
        stat_type: Which stat to analyze
        top_n: Number of top picks to return
        min_games: Minimum games needed for a player

    Returns:
        DataFrame with columns:
        player, prediction, line, direction, hit_prob,
        l10_rate, edge, confidence
    """
    stat_lower = stat_type.lower()
    pred_col = f"pred_{stat_lower}"

    if pred_col not in predictions_df.columns:
        print(f"Warning: {pred_col} not found in predictions")
        return pd.DataFrame()

    results = []

    for _, row in predictions_df.iterrows():
        player = row.get("player", row.get("PLAYER_NAME", "Unknown"))
        prediction = row.get(pred_col, 0)

        if prediction <= 0:
            continue

        # Get player's recent games
        player_games = game_logs_df[
            game_logs_df["PLAYER_NAME"].str.lower() == player.lower()
        ].sort_values("GAME_DATE", ascending=False)

        if len(player_games) < min_games:
            continue

        # Get recent stat values
        recent = player_games.head(10)
        if stat_type in recent.columns:
            recent_stats = recent[stat_type]
        elif "+" in stat_type:
            # Handle combo stats like PTS+REB
            parts = stat_type.split("+")
            recent_stats = sum(recent[p] for p in parts if p in recent.columns)
        else:
            continue

        # Find best line
        line, prob, direction = find_best_line(
            prediction, recent_stats, stat_type
        )

        if line is None:
            continue

        # Calculate historical hit rate
        l10_rate = calculate_historical_hit_rate(recent_stats, line, direction)

        # Get streak info
        streak = get_streak_info(recent_stats, line, direction)

        # Calculate edge
        edge = calculate_edge(prob)

        # Determine confidence
        if prob >= 0.65 and l10_rate >= 0.70:
            confidence = "high"
        elif prob >= 0.55 and l10_rate >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"

        results.append({
            "player": player,
            "stat": stat_type,
            "prediction": round(prediction, 1),
            "recent_avg": round(recent_stats.mean(), 1),
            "line": line,
            "direction": direction,
            "hit_prob": prob,
            "l10_rate": l10_rate,
            "edge": edge,
            "current_streak": streak["current_streak"],
            "confidence": confidence
        })

    # Create DataFrame and sort by hit probability
    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    # Sort by edge (best value) then by probability
    results_df = results_df.sort_values(
        ["edge", "hit_prob"],
        ascending=[False, False]
    ).head(top_n)

    return results_df


def generate_all_stats_props(
    predictors: dict,
    features_df: pd.DataFrame,
    game_logs_df: pd.DataFrame,
    players: list[str],
    top_n_per_stat: int = 10
) -> pd.DataFrame:
    """
    Generate best props across all stat types.

    Args:
        predictors: Dict of {stat: NBAPredictor} objects
        features_df: DataFrame with player features
        game_logs_df: Historical game logs
        players: List of players to analyze
        top_n_per_stat: How many picks per stat type

    Returns:
        Combined DataFrame of all best props
    """
    all_props = []

    for stat, predictor in predictors.items():
        # Get predictions for all players
        predictions = predictor.predict_batch(players, features_df)

        if predictions.empty:
            continue

        # Generate best props for this stat
        props = generate_best_props(
            predictions,
            game_logs_df,
            stat_type=stat,
            top_n=top_n_per_stat
        )

        all_props.append(props)

    if not all_props:
        return pd.DataFrame()

    combined = pd.concat(all_props, ignore_index=True)

    # Sort by edge across all stats
    combined = combined.sort_values("edge", ascending=False)

    return combined
