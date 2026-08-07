"""Build feature vectors for tonight's WNBA predictions.

Takes a player's recent-game feature row (from the engineered parquet) and
overrides the context-dependent fields to reflect **tonight's** opponent
instead of the player's most-recent game's opponent.

Fields adjusted for tonight:
- is_home (based on tonight's schedule)
- days_rest / is_back_to_back (based on gap between last game and tonight)
- opp_def_pts_allowed / opp_def_fg_pct / opp_def_fg3_pct (from team_def)
- opp_pts_def_rank / opp_ast_def_rank / opp_reb_def_rank (from def_vs_pos + player position)
- opp_elite_defense / opp_good_defense / opp_poor_defense (from opp_pts_def_rank)
- team_pace / opp_pace / game_pace (from team_stats)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

from utils.league_config import get_config


def build_tonight_feature_row(
    player_history: pd.DataFrame,
    tonight_opponent: str,
    is_home: bool,
    tonight_date: Optional[date] = None,
    team_def: Optional[pd.DataFrame] = None,
    def_vs_pos: Optional[pd.DataFrame] = None,
    team_stats: Optional[pd.DataFrame] = None,
    current_season: Optional[str] = None,
) -> dict:
    """Compose a feature dict for a player's *next* game.

    Args:
        player_history: enriched player rows (parquet), sorted by _date desc.
                        Uses iloc[0] for latest-game rolling averages.
        tonight_opponent: opponent team abbrev, e.g. "NYL"
        is_home: True if player's team is home tonight
        tonight_date: defaults to today's date
        team_def, def_vs_pos, team_stats: reference tables for context overrides
        current_season: e.g. "2026". Defaults to latest season in player_history.
    """
    if player_history.empty:
        return {}

    feat = player_history.iloc[0].to_dict()

    if tonight_date is None:
        tonight_date = date.today()
    if current_season is None:
        current_season = str(feat.get("SEASON", ""))

    # Home/away override
    feat["is_home"] = 1 if is_home else 0

    # Days rest / back-to-back override
    last_date = player_history.iloc[0].get("_date")
    if pd.notna(last_date):
        last_day = last_date.date() if hasattr(last_date, "date") else last_date
        rest = max(0, (tonight_date - last_day).days)
    else:
        rest = int(feat.get("days_rest", 2) or 2)
    feat["days_rest"] = rest
    feat["is_back_to_back"] = 1 if rest <= 1 else 0

    # Opponent defense stats
    if team_def is not None and not team_def.empty:
        opp_row = team_def[
            (team_def["TEAM_ABBREVIATION"] == tonight_opponent)
            & (team_def["SEASON"].astype(str) == str(current_season))
        ]
        if not opp_row.empty:
            r = opp_row.iloc[0]
            if "OPP_PTS" in r:
                feat["opp_def_pts_allowed"] = float(r.get("OPP_PTS", feat.get("opp_def_pts_allowed", 0)))
            if "OPP_FG_PCT" in r:
                feat["opp_def_fg_pct"] = float(r.get("OPP_FG_PCT", feat.get("opp_def_fg_pct", 0)))
            if "OPP_FG3_PCT" in r:
                feat["opp_def_fg3_pct"] = float(r.get("OPP_FG3_PCT", feat.get("opp_def_fg3_pct", 0)))

    # Opponent defense-vs-position ranks
    if def_vs_pos is not None and not def_vs_pos.empty:
        position = feat.get("POSITION", "F")
        dvp = def_vs_pos[
            (def_vs_pos["TEAM_ABBREVIATION"] == tonight_opponent)
            & (def_vs_pos["SEASON"].astype(str) == str(current_season))
            & (def_vs_pos["POSITION"] == position)
        ]
        if not dvp.empty:
            r = dvp.iloc[0]
            feat["opp_pts_def_rank"] = int(r.get("PTS_RANK", 7))
            feat["opp_ast_def_rank"] = int(r.get("AST_RANK", 7))
            feat["opp_reb_def_rank"] = int(r.get("REB_RANK", 7))
        else:
            feat.setdefault("opp_pts_def_rank", 7)
            feat.setdefault("opp_ast_def_rank", 7)
            feat.setdefault("opp_reb_def_rank", 7)
        # Rebuild the tier flags
        rank = feat["opp_pts_def_rank"]
        feat["opp_elite_defense"] = 1 if rank <= 4 else 0     # WNBA has ~13 teams, adjust tiers
        feat["opp_good_defense"] = 1 if 4 < rank <= 8 else 0
        feat["opp_poor_defense"] = 1 if rank > 8 else 0

    # Pace: player's team is either home or visitor; team_pace stays as-is
    if team_stats is not None and not team_stats.empty:
        opp_pace_row = team_stats[
            (team_stats["TEAM_ABBREVIATION"] == tonight_opponent)
            & (team_stats["SEASON"].astype(str) == str(current_season))
        ]
        if not opp_pace_row.empty and "PACE" in opp_pace_row.columns:
            opp_pace = float(opp_pace_row.iloc[0]["PACE"])
            feat["opp_pace"] = opp_pace
            team_pace = float(feat.get("team_pace", opp_pace))
            feat["game_pace"] = (team_pace + opp_pace) / 2

    return feat


def get_tonight_matchup_for_player(
    player_team: str,
    todays_games: list[dict],
) -> Optional[tuple[str, bool]]:
    """Return (opponent_abbrev, is_home) if player's team plays tonight, else None."""
    for g in todays_games:
        if g["home"]["abbrev"] == player_team:
            return g["away"]["abbrev"], True
        if g["away"]["abbrev"] == player_team:
            return g["home"]["abbrev"], False
    return None
