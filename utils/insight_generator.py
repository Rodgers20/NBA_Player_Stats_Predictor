"""
Player Insight Generator
========================
Generates contextual, human-readable insights for prop cards.

Each insight explains WHY a prop is interesting: recent form, matchup
quality, home/away split, blowout risk, and rest days.
"""

from datetime import datetime
import pandas as pd

STAT_LABELS = {
    "PTS": "points",
    "AST": "assists",
    "REB": "rebounds",
    "FG3M": "threes",
}

GRADE_THRESHOLDS = {
    # Defensive rank (1=best, 30=worst). Higher rank vs this position = easier matchup.
    "A": 24,   # rank 24-30 — very weak defense
    "B": 17,   # rank 17-23 — below-average defense
    "C": 10,   # rank 10-16 — average defense
    "D": 0,    # rank 1-9  — strong defense
}


def _matchup_grade(def_rank: int | None) -> str:
    if def_rank is None:
        return "C"
    if def_rank >= GRADE_THRESHOLDS["A"]:
        return "A"
    if def_rank >= GRADE_THRESHOLDS["B"]:
        return "B"
    if def_rank >= GRADE_THRESHOLDS["C"]:
        return "C"
    return "D"


def _trend(l5_avg: float, l10_avg: float) -> str:
    if l10_avg < 0.01:
        return "neutral"
    pct = (l5_avg - l10_avg) / l10_avg
    if pct > 0.10:
        return "hot"
    if pct < -0.10:
        return "cold"
    return "neutral"


def _check_blowout_risk(player_df: pd.DataFrame) -> bool:
    """
    Blowout risk: if the team has been losing by 10+ pts on average recently,
    star players tend to sit 4th-quarter garbage time.
    """
    if "PLUS_MINUS" not in player_df.columns or len(player_df) < 5:
        return False
    recent_pm = player_df.head(10)["PLUS_MINUS"].mean()
    return recent_pm < -10


def _rest_days(player_df: pd.DataFrame) -> int:
    if "_date" not in player_df.columns or len(player_df) < 1:
        return 1
    last_game = player_df["_date"].iloc[0]
    if pd.isna(last_game):
        return 1
    delta = (datetime.now().date() - last_game.date()).days
    return max(1, delta)


def _home_away_delta(player_df: pd.DataFrame, stat: str, is_home: bool) -> float:
    """Return how many more/fewer points the player averages at home vs away."""
    if stat not in player_df.columns or "MATCHUP" not in player_df.columns:
        return 0.0
    home_games = player_df[player_df["MATCHUP"].str.contains("vs.", na=False)].head(15)
    away_games = player_df[player_df["MATCHUP"].str.contains("@", na=False)].head(15)
    if home_games.empty or away_games.empty:
        return 0.0
    return round(home_games[stat].mean() - away_games[stat].mean(), 1)


def _build_factors(
    stat: str,
    l5_avg: float,
    line: float,
    trend: str,
    def_rank: int | None,
    blowout: bool,
    is_home: bool,
    location_delta: float,
    rest: int,
) -> list[dict]:
    factors = []

    # Trend
    if trend == "hot":
        factors.append({"text": f"Hot streak — {l5_avg:.1f} avg last 5 games", "positive": True})
    elif trend == "cold":
        factors.append({"text": f"Cold streak — {l5_avg:.1f} avg last 5 games", "positive": False})

    # Defensive matchup
    stat_label = STAT_LABELS.get(stat, stat.lower())
    if def_rank is not None:
        if def_rank >= 24:
            factors.append({"text": f"Weak defense — #{def_rank} vs this position", "positive": True})
        elif def_rank >= 18:
            factors.append({"text": f"Below-avg defense — #{def_rank} vs this position", "positive": True})
        elif def_rank <= 8:
            factors.append({"text": f"Tough defense — #{def_rank} vs this position", "positive": False})
        else:
            factors.append({"text": f"Defense ranks #{def_rank} vs this position", "positive": None})

    # Blowout risk
    if blowout:
        factors.append({"text": "Blowout risk — team allows big losses (may sit in 4th)", "positive": False})

    # Location
    if abs(location_delta) >= 1.5:
        location_word = "home" if is_home else "away"
        more_or_fewer = "more" if (is_home and location_delta > 0) or (not is_home and location_delta < 0) else "fewer"
        factors.append({
            "text": f"Averages {abs(location_delta):.1f} {more_or_fewer} {stat_label} {location_word}",
            "positive": more_or_fewer == "more",
        })

    # Rest days
    if rest >= 3:
        factors.append({"text": f"{rest} days rest — well-rested", "positive": True})
    elif rest == 1:
        factors.append({"text": "Back-to-back — played last night", "positive": False})

    # Projection vs line
    if l5_avg >= line + 3:
        factors.append({"text": f"Projection {l5_avg:.1f} well above line {line}", "positive": True})

    return factors


def _build_narrative(
    player_name: str,
    stat: str,
    l5_avg: float,
    trend: str,
    opponent: str,
    def_rank: int | None,
    blowout: bool,
    is_home: bool,
    location_delta: float,
) -> str:
    stat_label = STAT_LABELS.get(stat, stat.lower())
    location = "at home" if is_home else "on the road"

    # Trend opening
    if trend == "hot":
        opening = f"🔥 On a hot streak — averaging {l5_avg:.1f} {stat_label} over the last 5 games"
    elif trend == "cold":
        opening = f"Struggling recently — averaging {l5_avg:.1f} {stat_label} over the last 5 games"
    else:
        opening = f"Averaging {l5_avg:.1f} {stat_label} over the last 5 games"

    # Matchup context
    if def_rank is not None:
        if def_rank >= 24:
            matchup = f"facing a weak defense ({opponent} ranks #{def_rank} vs this position)"
        elif def_rank <= 8:
            matchup = f"facing a tough defense ({opponent} ranks #{def_rank} vs this position)"
        else:
            matchup = f"vs {opponent} (#{def_rank} defense vs this position)"
    else:
        matchup = f"vs {opponent}"

    # Location note
    loc_note = ""
    if abs(location_delta) >= 1.5:
        more_or_fewer = "more" if (is_home and location_delta > 0) or (not is_home and location_delta < 0) else "fewer"
        loc_note = f" Averages {abs(location_delta):.1f} {more_or_fewer} {stat_label} {location}."

    # Blowout warning
    blowout_note = " ⚠️ Blowout risk — opponent may blow out this team, limiting 4th-quarter minutes." if blowout else ""

    return f"{opening}, {matchup}.{loc_note}{blowout_note}"


def generate_player_insight(
    player_name: str,
    stat: str,
    line: float,
    opponent: str,
    player_df: pd.DataFrame,
    defense_vs_pos: pd.DataFrame,
    is_home: bool,
    position: str,
) -> dict:
    """
    Generate a contextual insight dict for a player prop card.

    Returns:
        {
            "narrative": str,          # Human-readable explanation
            "trend": "hot"|"cold"|"neutral",
            "blowout_risk": bool,
            "rest_days": int,
            "matchup_grade": "A"|"B"|"C"|"D",
            "def_rank": int|None,
            "factors": [{"text": str, "positive": bool|None}]
        }
    """
    if player_df.empty:
        return _empty_insight()

    sorted_df = player_df.sort_values("_date", ascending=False) if "_date" in player_df.columns else player_df

    recent_5 = sorted_df.head(5)
    recent_10 = sorted_df.head(10)

    l5_avg = recent_5[stat].mean() if stat in recent_5.columns else 0.0
    l10_avg = recent_10[stat].mean() if stat in recent_10.columns else 0.0

    trend = _trend(l5_avg, l10_avg)
    blowout = _check_blowout_risk(sorted_df)
    rest = _rest_days(sorted_df)
    location_delta = _home_away_delta(sorted_df, stat, is_home)

    # Defensive rank
    def_rank = None
    rank_col = "3PM_RANK" if stat == "FG3M" else f"{stat}_RANK"
    if not defense_vs_pos.empty and opponent:
        opp_def = defense_vs_pos[
            (defense_vs_pos["TEAM_ABBREVIATION"] == opponent) &
            (defense_vs_pos["POSITION"] == position)
        ]
        if not opp_def.empty and rank_col in opp_def.columns:
            def_rank = int(opp_def.iloc[0][rank_col])

    grade = _matchup_grade(def_rank)
    factors = _build_factors(stat, l5_avg, line, trend, def_rank, blowout, is_home, location_delta, rest)
    narrative = _build_narrative(player_name, stat, l5_avg, trend, opponent, def_rank, blowout, is_home, location_delta)

    return {
        "narrative": narrative,
        "trend": trend,
        "blowout_risk": blowout,
        "rest_days": rest,
        "matchup_grade": grade,
        "def_rank": def_rank,
        "l5_avg": round(l5_avg, 1),
        "l10_avg": round(l10_avg, 1),
        "factors": factors,
    }


def _empty_insight() -> dict:
    return {
        "narrative": "",
        "trend": "neutral",
        "blowout_risk": False,
        "rest_days": 1,
        "matchup_grade": "C",
        "def_rank": None,
        "l5_avg": 0.0,
        "l10_avg": 0.0,
        "factors": [],
    }
