# utils/prop_scorer.py
"""
Smart Prop Scoring Module
=========================
Multi-factor scoring system that considers:
- Player role (starter/rotation/bench)
- Minutes opportunity
- Team injury context
- Defense matchup IN CONTEXT
- Recent form
- Home/away
- H2H history
- Rest/back-to-back

This replaces the oversimplified "defense rank #27" analysis
with comprehensive reasoning that shows BOTH positive and negative factors.
"""

import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats


# =============================================================================
# SCORING WEIGHTS
# =============================================================================

SCORING_WEIGHTS = {
    "base_probability": 0.35,    # Model prediction / historical hit rate
    "opportunity": 0.25,          # Role + minutes (CRITICAL for bench players)
    "form": 0.15,                 # Hot/cold streaks
    "matchup": 0.10,              # Defense ranking (reduced from over-reliance)
    "home_away": 0.05,            # Home court advantage
    "h2h": 0.05,                  # Head-to-head history
    "rest": 0.05,                 # Back-to-back, rest days
}


# =============================================================================
# ROLE DETECTION
# =============================================================================

def detect_player_role(player_df: pd.DataFrame) -> dict:
    """
    Detect if player is STARTER, ROTATION, or BENCH based on minutes patterns.

    Role classification:
    - STARTER: 28+ minutes avg (primary player)
    - ROTATION: 18-28 minutes avg (consistent rotation)
    - BENCH: <18 minutes avg (limited, variable)

    Args:
        player_df: Player's game log DataFrame (sorted by date descending)

    Returns:
        dict with role, avg_minutes, minutes_consistency, minutes_trend
    """
    if len(player_df) < 3:
        return {
            "role": "UNKNOWN",
            "avg_minutes": 0,
            "minutes_consistency": 0,
            "minutes_trend": 0,
            "games_analyzed": 0
        }

    recent = player_df.head(10)

    if "MIN" not in recent.columns:
        return {
            "role": "UNKNOWN",
            "avg_minutes": 0,
            "minutes_consistency": 0,
            "minutes_trend": 0,
            "games_analyzed": len(recent)
        }

    # Handle MIN column that might be string format "MM:SS" or numeric
    try:
        if recent["MIN"].dtype == object:
            # Convert "MM:SS" to minutes
            def parse_min(x):
                if pd.isna(x):
                    return 0
                if isinstance(x, str) and ":" in x:
                    parts = x.split(":")
                    return int(parts[0]) + int(parts[1]) / 60
                return float(x) if x else 0
            minutes = recent["MIN"].apply(parse_min)
        else:
            minutes = recent["MIN"].fillna(0)
    except Exception:
        minutes = pd.Series([20] * len(recent))  # Default fallback

    avg_min = minutes.mean()
    min_std = minutes.std() if len(minutes) > 1 else 0

    # Role classification
    if avg_min >= 28:
        role = "STARTER"
    elif avg_min >= 18:
        role = "ROTATION"
    else:
        role = "BENCH"

    # Minutes consistency (0-1, higher = more consistent)
    consistency = max(0, 1 - (min_std / avg_min)) if avg_min > 0 else 0

    # Minutes trend (positive = getting more time)
    l5_min = minutes.head(5).mean() if len(minutes) >= 5 else avg_min
    l10_min = minutes.mean()
    minutes_trend = (l5_min - l10_min) / l10_min if l10_min > 0 else 0

    return {
        "role": role,
        "avg_minutes": round(avg_min, 1),
        "minutes_consistency": round(consistency, 2),
        "minutes_trend": round(minutes_trend, 3),
        "games_analyzed": len(recent)
    }


# =============================================================================
# OPPORTUNITY CONTEXT
# =============================================================================

def assess_opportunity_context(
    player_name: str,
    player_team: str,
    player_role: str,
    player_position: str,
    injury_checker=None
) -> dict:
    """
    Assess how team context affects this player's opportunity.

    Key insight: A bench player on a fully healthy team gets LIMITED minutes,
    regardless of how bad the opponent's defense is.

    Args:
        player_name: Player name
        player_team: Team abbreviation
        player_role: STARTER/ROTATION/BENCH from detect_player_role
        player_position: G/F/C
        injury_checker: Optional function to check injuries

    Returns:
        dict with opportunity_modifier, boost_reason, risk_reason
    """
    opportunity_modifier = 0
    boost_reasons = []
    risk_reasons = []

    # Base opportunity by role
    role_base = {
        "STARTER": 1.0,
        "ROTATION": 0.85,
        "BENCH": 0.65,
        "UNKNOWN": 0.75
    }
    base = role_base.get(player_role, 0.75)

    # For bench players, add inherent risk warning
    if player_role == "BENCH":
        risk_reasons.append({
            "type": "role",
            "reason": "Bench role - limited and variable minutes",
            "impact": -0.10
        })
        opportunity_modifier -= 0.10

    # Check for injury context if checker provided
    if injury_checker is not None:
        try:
            # Check if key teammates are out (opportunity boost)
            # or if player returning from injury (risk)
            injury_info = injury_checker(player_name)

            if injury_info:
                status = injury_info.get("status", "")
                if status == "QUESTIONABLE":
                    risk_reasons.append({
                        "type": "injury",
                        "reason": "Questionable status - may have minutes limit",
                        "impact": -0.08
                    })
                    opportunity_modifier -= 0.08
                elif status == "PROBABLE":
                    # Slight concern but usually plays
                    pass
        except Exception:
            pass

    return {
        "base_opportunity": base,
        "opportunity_modifier": opportunity_modifier,
        "final_opportunity": max(0.3, base + opportunity_modifier),
        "boost_reasons": boost_reasons,
        "risk_reasons": risk_reasons
    }


# =============================================================================
# DEFENSE RANKING IN CONTEXT
# =============================================================================

def contextualize_defense_ranking(
    def_rank: int,
    player_role: str,
    stat_type: str = "PTS"
) -> dict:
    """
    Convert raw defensive ranking into contextual impact score.

    CRITICAL: Rank #27 doesn't mean +27% for a bench player!
    A bench player needs MINUTES first, then defense matters.

    Args:
        def_rank: 1-30 ranking (1 = best defense, 30 = worst)
        player_role: STARTER/ROTATION/BENCH
        stat_type: PTS/AST/REB

    Returns:
        dict with raw_rank, adjusted_impact, tier, reasoning
    """
    # Base impact calculation
    # 1-10: Elite defense (-10% to -5%)
    # 11-20: Average defense (neutral)
    # 21-30: Poor defense (+5% to +15%)

    if def_rank <= 6:
        base_impact = -0.10
        tier = "elite"
    elif def_rank <= 10:
        base_impact = -0.05
        tier = "good"
    elif def_rank <= 20:
        base_impact = 0.0
        tier = "average"
    elif def_rank <= 25:
        base_impact = 0.05
        tier = "below_average"
    else:
        base_impact = min(0.10 + (def_rank - 25) * 0.01, 0.15)
        tier = "poor"

    # CRITICAL: Adjust impact by player role
    # Bench players need minutes first - defense matters less
    role_multiplier = {
        "STARTER": 1.0,      # Full effect
        "ROTATION": 0.7,     # Moderate effect
        "BENCH": 0.3,        # Minimal effect - minutes matter more
        "UNKNOWN": 0.5
    }

    multiplier = role_multiplier.get(player_role, 0.5)
    adjusted_impact = base_impact * multiplier

    # Build reasoning
    tier_text = {
        "elite": f"Elite defense (#{def_rank})",
        "good": f"Good defense (#{def_rank})",
        "average": f"Average defense (#{def_rank})",
        "below_average": f"Below avg defense (#{def_rank})",
        "poor": f"Poor defense (#{def_rank})"
    }

    if player_role == "BENCH" and tier in ["below_average", "poor"]:
        reasoning = f"{tier_text[tier]} - modest boost (bench player)"
    elif tier in ["elite", "good"]:
        reasoning = f"{tier_text[tier]} - tough matchup"
    elif tier in ["below_average", "poor"]:
        reasoning = f"{tier_text[tier]} - favorable matchup"
    else:
        reasoning = f"{tier_text[tier]} - neutral"

    return {
        "raw_rank": def_rank,
        "base_impact": base_impact,
        "adjusted_impact": adjusted_impact,
        "tier": tier,
        "reasoning": reasoning,
        "role_multiplier": multiplier
    }


# =============================================================================
# FORM ANALYSIS
# =============================================================================

def analyze_form(player_df: pd.DataFrame, stat_cols: list) -> dict:
    """
    Analyze player's recent form vs historical performance.

    Args:
        player_df: Player game logs (sorted by date descending)
        stat_cols: List of stat columns to sum (e.g., ["PTS"] or ["PTS", "AST"])

    Returns:
        dict with form_score, form_type, reasoning
    """
    if len(player_df) < 5:
        return {
            "form_score": 0,
            "form_type": "unknown",
            "reasoning": None,
            "l5_avg": 0,
            "l10_avg": 0,
            "season_avg": 0
        }

    # Calculate stat values
    available_cols = [c for c in stat_cols if c in player_df.columns]
    if not available_cols:
        return {
            "form_score": 0,
            "form_type": "unknown",
            "reasoning": None,
            "l5_avg": 0,
            "l10_avg": 0,
            "season_avg": 0
        }

    values = player_df[available_cols].sum(axis=1)
    l5_avg = values.head(5).mean()
    l10_avg = values.head(10).mean()
    season_avg = values.mean()

    # Determine form
    if l5_avg > season_avg * 1.15:
        form_score = 0.10
        form_type = "hot"
        reasoning = f"Hot streak: L5 avg {l5_avg:.1f} >> season {season_avg:.1f}"
    elif l5_avg > l10_avg * 1.08:
        form_score = 0.05
        form_type = "trending_up"
        reasoning = f"Trending up: L5 {l5_avg:.1f} > L10 {l10_avg:.1f}"
    elif l5_avg < season_avg * 0.85:
        form_score = -0.10
        form_type = "cold"
        reasoning = f"Cold: L5 avg {l5_avg:.1f} << season {season_avg:.1f}"
    elif l5_avg < l10_avg * 0.92:
        form_score = -0.05
        form_type = "trending_down"
        reasoning = f"Trending down: L5 {l5_avg:.1f} < L10 {l10_avg:.1f}"
    else:
        form_score = 0
        form_type = "stable"
        reasoning = None

    return {
        "form_score": form_score,
        "form_type": form_type,
        "reasoning": reasoning,
        "l5_avg": round(l5_avg, 1),
        "l10_avg": round(l10_avg, 1),
        "season_avg": round(season_avg, 1)
    }


# =============================================================================
# HOME/AWAY ANALYSIS
# =============================================================================

def analyze_home_away(player_df: pd.DataFrame, stat_cols: list, is_home: bool) -> dict:
    """
    Analyze player's home vs away performance.

    Args:
        player_df: Player game logs
        stat_cols: Stat columns to analyze
        is_home: Whether today's game is at home

    Returns:
        dict with home_away_score, reasoning
    """
    if "MATCHUP" not in player_df.columns or len(player_df) < 10:
        return {
            "home_away_score": 0,
            "reasoning": None,
            "home_avg": 0,
            "away_avg": 0
        }

    available_cols = [c for c in stat_cols if c in player_df.columns]
    if not available_cols:
        return {
            "home_away_score": 0,
            "reasoning": None,
            "home_avg": 0,
            "away_avg": 0
        }

    # Split by home/away
    home_mask = player_df["MATCHUP"].str.contains("vs.", na=False)
    home_games = player_df[home_mask]
    away_games = player_df[~home_mask]

    if len(home_games) < 3 or len(away_games) < 3:
        return {
            "home_away_score": 0.01 if is_home else 0,
            "reasoning": "Playing at home" if is_home else None,
            "home_avg": 0,
            "away_avg": 0
        }

    home_avg = home_games[available_cols].sum(axis=1).mean()
    away_avg = away_games[available_cols].sum(axis=1).mean()

    score = 0
    reasoning = None

    if is_home:
        if home_avg > away_avg * 1.08:
            score = 0.05
            reasoning = f"Strong at home: {home_avg:.1f} vs {away_avg:.1f} away"
        else:
            score = 0.01
            reasoning = "Playing at home"
    else:
        if away_avg > home_avg * 1.05:
            score = 0.02
            reasoning = f"Good on road: {away_avg:.1f} away"
        elif away_avg < home_avg * 0.90:
            score = -0.03
            reasoning = f"Struggles on road: {away_avg:.1f} away vs {home_avg:.1f} home"

    return {
        "home_away_score": score,
        "reasoning": reasoning,
        "home_avg": round(home_avg, 1),
        "away_avg": round(away_avg, 1)
    }


# =============================================================================
# HEAD-TO-HEAD ANALYSIS
# =============================================================================

def analyze_h2h(player_df: pd.DataFrame, opponent: str, stat_cols: list, line: float) -> dict:
    """
    Analyze player's history vs specific opponent.

    Args:
        player_df: Player game logs
        opponent: Opponent team abbreviation
        stat_cols: Stat columns
        line: Prop line to check

    Returns:
        dict with h2h_score, hit_rate, avg, reasoning
    """
    if not opponent or "MATCHUP" not in player_df.columns:
        return {
            "h2h_score": 0,
            "reasoning": None,
            "h2h_games": 0,
            "h2h_avg": 0,
            "h2h_hit_rate": 0
        }

    # Filter games vs this opponent
    h2h_mask = player_df["MATCHUP"].str.contains(opponent, na=False)
    h2h_games = player_df[h2h_mask]

    if len(h2h_games) < 3:
        return {
            "h2h_score": 0,
            "reasoning": None,
            "h2h_games": len(h2h_games),
            "h2h_avg": 0,
            "h2h_hit_rate": 0
        }

    available_cols = [c for c in stat_cols if c in h2h_games.columns]
    if not available_cols:
        return {
            "h2h_score": 0,
            "reasoning": None,
            "h2h_games": len(h2h_games),
            "h2h_avg": 0,
            "h2h_hit_rate": 0
        }

    h2h_values = h2h_games[available_cols].sum(axis=1)
    h2h_avg = h2h_values.mean()
    h2h_hit_rate = (h2h_values >= line).sum() / len(h2h_values)

    score = 0
    reasoning = None

    if h2h_hit_rate >= 0.75:
        score = 0.08
        reasoning = f"Dominates {opponent}: {h2h_hit_rate:.0%} hit rate ({len(h2h_games)} games)"
    elif h2h_hit_rate >= 0.60:
        score = 0.04
        reasoning = f"Good vs {opponent}: {h2h_hit_rate:.0%} hit rate"
    elif h2h_hit_rate <= 0.30:
        score = -0.06
        reasoning = f"Struggles vs {opponent}: {h2h_hit_rate:.0%} hit rate"
    elif h2h_hit_rate <= 0.40:
        score = -0.03
        reasoning = f"Below avg vs {opponent}: {h2h_hit_rate:.0%} hit rate"

    return {
        "h2h_score": score,
        "reasoning": reasoning,
        "h2h_games": len(h2h_games),
        "h2h_avg": round(h2h_avg, 1),
        "h2h_hit_rate": round(h2h_hit_rate, 2)
    }


# =============================================================================
# REST / SCHEDULE ANALYSIS
# =============================================================================

def analyze_rest(player_df: pd.DataFrame) -> dict:
    """
    Analyze rest days and back-to-back impact.

    Args:
        player_df: Player game logs (sorted by date descending)

    Returns:
        dict with rest_score, reasoning
    """
    if "_date" not in player_df.columns or len(player_df) < 2:
        return {
            "rest_score": 0,
            "reasoning": None,
            "days_rest": None,
            "is_b2b": False
        }

    # Calculate days since last game
    try:
        last_game = player_df["_date"].iloc[0]
        prev_game = player_df["_date"].iloc[1]
        days_rest = (last_game - prev_game).days if pd.notna(last_game) and pd.notna(prev_game) else 2
    except Exception:
        days_rest = 2  # Default

    is_b2b = days_rest <= 1

    if is_b2b:
        return {
            "rest_score": -0.05,
            "reasoning": "Back-to-back game - fatigue factor",
            "days_rest": days_rest,
            "is_b2b": True
        }
    elif days_rest >= 4:
        return {
            "rest_score": 0.03,
            "reasoning": f"{days_rest} days rest - well rested",
            "days_rest": days_rest,
            "is_b2b": False
        }
    else:
        return {
            "rest_score": 0,
            "reasoning": None,
            "days_rest": days_rest,
            "is_b2b": False
        }


# =============================================================================
# BASE PROBABILITY CALCULATION
# =============================================================================

def calculate_base_probability(
    prediction: float,
    line: float,
    std_dev: float,
    direction: str = "over"
) -> float:
    """
    Calculate base hit probability using normal distribution.

    Args:
        prediction: Model prediction or average
        line: Prop line
        std_dev: Standard deviation
        direction: "over" or "under"

    Returns:
        Probability (0-1)
    """
    if std_dev <= 0:
        std_dev = prediction * 0.2 if prediction > 0 else 1

    if direction == "over":
        # P(X >= line) = 1 - CDF(line)
        prob = 1 - stats.norm.cdf(line, loc=prediction, scale=std_dev)
    else:
        # P(X <= line) = CDF(line)
        prob = stats.norm.cdf(line, loc=prediction, scale=std_dev)

    return max(0.05, min(0.95, prob))


# =============================================================================
# MAIN COMPREHENSIVE SCORING FUNCTION
# =============================================================================

def calculate_smart_prop_score(
    player_name: str,
    stat_cols: list,
    line: float,
    player_df: pd.DataFrame,
    info: dict,
    defense_data: pd.DataFrame = None,
    prediction: float = None,
    injury_checker=None
) -> dict:
    """
    Calculate comprehensive prop score with full context.

    This is the MAIN function that replaces the oversimplified scoring.
    It considers role, minutes, matchup IN CONTEXT, form, and more.

    Args:
        player_name: Player name
        stat_cols: List of stat columns (e.g., ["PTS"] or ["PTS", "AST"])
        line: Prop line
        player_df: Player's game logs (sorted by date descending)
        info: Dict with team, opponent, position, is_home
        defense_data: DEFENSE_VS_POS DataFrame
        prediction: Model prediction (optional, uses L10 avg if None)
        injury_checker: Optional function to check injuries

    Returns:
        Comprehensive analysis dict with score, factors, reasoning
    """
    available_cols = [c for c in stat_cols if c in player_df.columns]
    if not available_cols or len(player_df) < 5:
        return {
            "final_score": 0.5,
            "confidence": "LOW",
            "positive_factors": [],
            "negative_factors": [],
            "role": "UNKNOWN",
            "avg_minutes": 0
        }

    recent = player_df.head(10)
    values = recent[available_cols].sum(axis=1)

    # =========================================
    # FACTOR 1: BASE PROBABILITY
    # =========================================
    if prediction is None:
        prediction = values.mean()

    std_dev = values.std() if len(values) > 1 else prediction * 0.2
    base_prob = calculate_base_probability(prediction, line, std_dev, "over")

    # Historical hit rate
    l10_rate = (values >= line).sum() / len(values) if len(values) > 0 else 0.5
    l5_rate = (values.head(5) >= line).sum() / 5 if len(values) >= 5 else l10_rate

    # =========================================
    # FACTOR 2: PLAYER ROLE & OPPORTUNITY
    # =========================================
    role_data = detect_player_role(player_df)
    opportunity_data = assess_opportunity_context(
        player_name,
        info.get("team", ""),
        role_data["role"],
        info.get("position", "F"),
        injury_checker
    )

    # =========================================
    # FACTOR 3: MATCHUP (Defense Ranking)
    # =========================================
    matchup_data = {"adjusted_impact": 0, "reasoning": None, "raw_rank": 15}

    if defense_data is not None and not defense_data.empty and info.get("opponent"):
        opponent = info.get("opponent", "")
        position = info.get("position", "F")

        # Normalize position
        if "G" in str(position):
            pos_filter = "G"
        elif "C" in str(position):
            pos_filter = "C"
        else:
            pos_filter = "F"

        opp_def = defense_data[
            (defense_data["TEAM_ABBREVIATION"] == opponent) &
            (defense_data["POSITION"] == pos_filter)
        ]

        if len(opp_def) > 0:
            # Get relevant rank based on stat type
            stat_key = stat_cols[0] if stat_cols else "PTS"
            rank_col = f"{stat_key}_RANK" if f"{stat_key}_RANK" in opp_def.columns else "PTS_RANK"
            def_rank = int(opp_def[rank_col].iloc[0]) if rank_col in opp_def.columns else 15

            matchup_data = contextualize_defense_ranking(
                def_rank,
                role_data["role"],
                stat_key
            )

    # =========================================
    # FACTOR 4: RECENT FORM
    # =========================================
    form_data = analyze_form(player_df, stat_cols)

    # =========================================
    # FACTOR 5: HOME/AWAY
    # =========================================
    home_away_data = analyze_home_away(
        player_df,
        stat_cols,
        info.get("is_home", False)
    )

    # =========================================
    # FACTOR 6: HEAD-TO-HEAD
    # =========================================
    h2h_data = analyze_h2h(
        player_df,
        info.get("opponent", ""),
        stat_cols,
        line
    )

    # =========================================
    # FACTOR 7: REST/SCHEDULE
    # =========================================
    rest_data = analyze_rest(player_df)

    # =========================================
    # CALCULATE WEIGHTED FINAL SCORE
    # =========================================

    # Normalize components to 0-1 scale
    prob_component = base_prob  # Already 0-1
    opportunity_component = opportunity_data["final_opportunity"]  # 0.3-1.0
    matchup_component = 0.5 + matchup_data["adjusted_impact"]  # 0.35-0.65
    form_component = 0.5 + form_data["form_score"]  # 0.4-0.6
    home_component = 0.5 + home_away_data["home_away_score"]  # 0.47-0.55
    h2h_component = 0.5 + h2h_data["h2h_score"]  # 0.44-0.58
    rest_component = 0.5 + rest_data["rest_score"]  # 0.45-0.53

    # Weighted sum
    final_score = (
        prob_component * SCORING_WEIGHTS["base_probability"] +
        opportunity_component * SCORING_WEIGHTS["opportunity"] +
        matchup_component * SCORING_WEIGHTS["matchup"] +
        form_component * SCORING_WEIGHTS["form"] +
        home_component * SCORING_WEIGHTS["home_away"] +
        h2h_component * SCORING_WEIGHTS["h2h"] +
        rest_component * SCORING_WEIGHTS["rest"]
    )

    # Normalize to reasonable range
    final_score = max(0.15, min(0.90, final_score))

    # =========================================
    # BUILD POSITIVE & NEGATIVE FACTORS
    # =========================================
    positive_factors = []
    negative_factors = []

    # Form factors
    if form_data["reasoning"]:
        if form_data["form_score"] > 0:
            positive_factors.append({
                "type": "form",
                "reason": form_data["reasoning"],
                "impact": form_data["form_score"]
            })
        else:
            negative_factors.append({
                "type": "form",
                "reason": form_data["reasoning"],
                "impact": form_data["form_score"]
            })

    # Matchup factors
    if matchup_data["reasoning"]:
        if matchup_data["adjusted_impact"] > 0.02:
            positive_factors.append({
                "type": "matchup",
                "reason": matchup_data["reasoning"],
                "impact": matchup_data["adjusted_impact"]
            })
        elif matchup_data["adjusted_impact"] < -0.02:
            negative_factors.append({
                "type": "matchup",
                "reason": matchup_data["reasoning"],
                "impact": matchup_data["adjusted_impact"]
            })

    # Opportunity/role factors
    for risk in opportunity_data["risk_reasons"]:
        negative_factors.append(risk)
    for boost in opportunity_data["boost_reasons"]:
        positive_factors.append(boost)

    # Home/away factors
    if home_away_data["reasoning"]:
        if home_away_data["home_away_score"] > 0:
            positive_factors.append({
                "type": "home_away",
                "reason": home_away_data["reasoning"],
                "impact": home_away_data["home_away_score"]
            })
        elif home_away_data["home_away_score"] < 0:
            negative_factors.append({
                "type": "home_away",
                "reason": home_away_data["reasoning"],
                "impact": home_away_data["home_away_score"]
            })

    # H2H factors
    if h2h_data["reasoning"]:
        if h2h_data["h2h_score"] > 0:
            positive_factors.append({
                "type": "h2h",
                "reason": h2h_data["reasoning"],
                "impact": h2h_data["h2h_score"]
            })
        else:
            negative_factors.append({
                "type": "h2h",
                "reason": h2h_data["reasoning"],
                "impact": h2h_data["h2h_score"]
            })

    # Rest factors
    if rest_data["reasoning"]:
        if rest_data["rest_score"] > 0:
            positive_factors.append({
                "type": "rest",
                "reason": rest_data["reasoning"],
                "impact": rest_data["rest_score"]
            })
        else:
            negative_factors.append({
                "type": "rest",
                "reason": rest_data["reasoning"],
                "impact": rest_data["rest_score"]
            })

    # Hit rate factors
    if l10_rate >= 0.70:
        positive_factors.append({
            "type": "consistency",
            "reason": f"Consistent: {l10_rate:.0%} hit rate L10",
            "impact": 0.05
        })
    elif l10_rate <= 0.30:
        negative_factors.append({
            "type": "consistency",
            "reason": f"Inconsistent: {l10_rate:.0%} hit rate L10",
            "impact": -0.05
        })

    # Sort by impact
    positive_factors.sort(key=lambda x: x["impact"], reverse=True)
    negative_factors.sort(key=lambda x: x["impact"])

    # =========================================
    # DETERMINE CONFIDENCE
    # =========================================
    if role_data["role"] == "STARTER" and role_data["minutes_consistency"] > 0.6:
        if final_score >= 0.65:
            confidence = "HIGH"
        elif final_score >= 0.50:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    elif role_data["role"] == "BENCH":
        confidence = "LOW"  # Always low for bench players
    else:
        if final_score >= 0.60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

    return {
        "player": player_name,
        "stat": "+".join(stat_cols),
        "line": line,
        "final_score": round(final_score, 3),
        "final_percentage": f"{final_score * 100:.0f}%",

        # Component data
        "base_probability": round(base_prob, 3),
        "l10_rate": round(l10_rate, 2),
        "l5_rate": round(l5_rate, 2),
        "prediction": round(prediction, 1),

        # Role & opportunity
        "role": role_data["role"],
        "avg_minutes": role_data["avg_minutes"],
        "minutes_consistency": role_data["minutes_consistency"],

        # Form
        "l5_avg": form_data["l5_avg"],
        "l10_avg": form_data["l10_avg"],
        "season_avg": form_data["season_avg"],

        # Matchup
        "def_rank": matchup_data["raw_rank"],
        "matchup_tier": matchup_data.get("tier", "average"),

        # Factors
        "positive_factors": positive_factors[:4],
        "negative_factors": negative_factors[:4],

        # Confidence
        "confidence": confidence
    }


# =============================================================================
# SMART REASONING GENERATOR
# =============================================================================

def generate_smart_reasoning(score_data: dict) -> list[str]:
    """
    Generate human-readable reasoning with appropriate icons.

    Shows BOTH positive and negative factors for balanced analysis.

    Args:
        score_data: Output from calculate_smart_prop_score

    Returns:
        List of formatted reason strings
    """
    reasons = []

    # Add positive factors
    for factor in score_data.get("positive_factors", [])[:3]:
        icon = {
            "form": "+",
            "matchup": "+",
            "home_away": "+",
            "h2h": "+",
            "rest": "+",
            "consistency": "+",
            "opportunity": "+"
        }.get(factor["type"], "+")
        reasons.append(f"{icon} {factor['reason']}")

    # Add negative factors
    for factor in score_data.get("negative_factors", [])[:3]:
        icon = {
            "form": "-",
            "matchup": "-",
            "role": "!",
            "injury": "!",
            "home_away": "-",
            "h2h": "-",
            "rest": "-",
            "consistency": "-"
        }.get(factor["type"], "-")
        reasons.append(f"{icon} {factor['reason']}")

    return reasons[:5]
