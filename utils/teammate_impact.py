# utils/teammate_impact.py
"""
Teammate Impact Analysis Module
===============================
Analyzes how a player's stats change when a specific teammate is OUT.
"""

import pandas as pd
from typing import Optional

def analyze_teammate_impact(
    target_player: str,
    teammate_out: str,
    target_player_games: pd.DataFrame,
    teammate_games: pd.DataFrame,
    stat: str = "PTS"
) -> Optional[dict]:
    """
    Calculate stats for `target_player` when `teammate_out` is missing.
    
    Args:
        target_player: Name of player to analyze (e.g. "LeBron James")
        teammate_out: Name of missing teammate (e.g. "Anthony Davis")
        target_player_games: DataFrame of target player's logs
        teammate_games: DataFrame of teammate's logs
        stat: Stat to analyze (PTS, AST, REB)
        
    Returns:
        dict with impact analysis or None if not enough data
    """
    if len(target_player_games) < 10 or len(teammate_games) < 10:
        return None
        
    # Standardize dates
    if "_date" not in target_player_games.columns:
        target_player_games = target_player_games.copy()
        target_player_games["_date"] = pd.to_datetime(target_player_games["GAME_DATE"], format="%b %d, %Y", errors="coerce")
        
    if "_date" not in teammate_games.columns:
        teammate_games = teammate_games.copy()
        teammate_games["_date"] = pd.to_datetime(teammate_games["GAME_DATE"], format="%b %d, %Y", errors="coerce")

    # Find games where teammate played significant minutes
    teammate_active_dates = set(teammate_games[teammate_games["MIN"] > 15]["_date"].dt.date)
    
    # Filter target player games
    games_with = []
    games_without = []
    
    for _, row in target_player_games.iterrows():
        game_date = row["_date"].date()
        val = row.get(stat, 0)
        
        if game_date in teammate_active_dates:
            games_with.append(val)
        else:
            games_without.append(val)
            
    if len(games_without) < 3: # Need minimal sample
        return None
        
    avg_with = sum(games_with) / len(games_with) if games_with else 0
    avg_without = sum(games_without) / len(games_without)
    diff = avg_without - avg_with
    
    return {
        "teammate": teammate_out,
        "avg_with": round(avg_with, 1),
        "avg_without": round(avg_without, 1),
        "diff": round(diff, 1),
        "games_without": len(games_without),
        "stat": stat
    }

def get_star_impact_statement(
    player_name: str,
    player_team: str,
    game_logs_df: pd.DataFrame, 
    unavailable_players: list,
    stat: str = "PTS"
) -> Optional[str]:
    """
    Generate a statement like "LeBron averages 28.5 PTS without AD".
    
    Args:
        player_name: Target player
        player_team: Team abbreviation
        game_logs_df: All game logs
        unavailable_players: List of players marked OUT today
        stat: Stat to check
    """
    # 1. Identify teammates
    # This is rough, usually we'd have a roster. 
    # For now, we assume `unavailable_players` contains relevant names.
    
    player_games = game_logs_df[game_logs_df["PLAYER_NAME"] == player_name]
    if player_games.empty: return None
    
    impacts = []
    
    for teammate in unavailable_players:
        if teammate == player_name: continue
        
        # Check if they are on same team (rough check)
        tm_games = game_logs_df[game_logs_df["PLAYER_NAME"] == teammate]
        if tm_games.empty: continue
        
        # Check if they played for the same team recently
        last_tm_game = tm_games.sort_values("GAME_DATE").iloc[-1]
        tm_team = last_tm_game.get("TEAM_ABBREVIATION")
        
        if tm_team != player_team: continue
        
        # Analyze
        impact = analyze_teammate_impact(player_name, teammate, player_games, tm_games, stat)
        
        if impact and impact["diff"] > 1.5: # Only show significant positive impact
            impacts.append(impact)
            
    if not impacts: return None
    
    # Take the most impactful one
    best = max(impacts, key=lambda x: x["diff"])
    
    return f"{player_name} averages {best['avg_without']} {stat} without {best['teammate']} (vs {best['avg_with']} with)."
