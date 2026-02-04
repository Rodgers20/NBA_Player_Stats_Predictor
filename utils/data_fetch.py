# utils/data_fetch.py
"""
NBA Data Fetching Module
========================
This module handles all communication with the NBA Stats API.

KEY CONCEPTS:
-------------
1. The NBA API uses numeric IDs for players and teams (not names)
   - "LeBron James" → player_id = 2544
   - "Los Angeles Lakers" → team_id = 1610612747

2. Seasons are formatted as "YYYY-YY" (e.g., "2024-25" for the 2024-2025 season)

3. The API returns data in JSON format, which we convert to pandas DataFrames
   for easier manipulation and analysis.

4. Rate limiting: The NBA API can block requests if you hit it too fast.
   We add small delays between requests to be respectful.
"""

import time
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashteamstats,
    playergamelog,
    playerdashboardbygeneralsplits,  # For home/away, wins/losses splits
    leaguedashplayerstats,  # For league-wide player stats
)
from nba_api.stats.static import players, teams


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_player_id(player_name: str) -> int | None:
    """
    Convert a player's name to their NBA ID.

    WHY THIS EXISTS:
    The NBA API requires numeric IDs, not names. This function handles the lookup.

    Args:
        player_name: Full name like "LeBron James" or "Stephen Curry"

    Returns:
        The player's NBA ID (int) or None if not found

    Example:
        >>> get_player_id("LeBron James")
        2544
    """
    # players.find_players_by_full_name() searches a local database
    # that comes bundled with the nba_api package
    player_dict = players.find_players_by_full_name(player_name)

    if not player_dict:
        print(f"Player '{player_name}' not found.")
        return None

    # The function returns a list (in case of multiple matches)
    # We take the first match's ID
    return player_dict[0]["id"]


def get_team_id(team_name: str) -> int | None:
    """
    Convert a team's name to their NBA ID.

    Args:
        team_name: Can be full name ("Los Angeles Lakers"),
                   city ("Los Angeles"), or abbreviation ("LAL")

    Returns:
        The team's NBA ID or None if not found
    """
    all_teams = teams.get_teams()

    # Search by full name, city, or abbreviation
    for team in all_teams:
        if (team_name.lower() in team["full_name"].lower() or
            team_name.lower() == team["abbreviation"].lower()):
            return team["id"]

    print(f"Team '{team_name}' not found.")
    return None


# =============================================================================
# PLAYER DATA FUNCTIONS
# =============================================================================

def get_player_stats(
    player_name: str,
    season: str = "2024-25",
    season_type: str = "Regular Season"
) -> pd.DataFrame:
    """
    Fetch game-by-game stats for a player in a specific season.

    WHAT THIS RETURNS:
    Each row = one game the player played. Columns include:
    - GAME_DATE: When the game was played
    - MATCHUP: Who they played (e.g., "LAL vs. GSW" or "LAL @ GSW")
    - PTS, AST, REB: Points, assists, rebounds
    - FGM, FGA, FG_PCT: Field goals made, attempted, percentage
    - MIN: Minutes played
    - PLUS_MINUS: Point differential while player was on court

    WHY GAME-BY-GAME DATA:
    For ML predictions, we need individual game data (not season averages) because:
    1. We can calculate rolling averages (last 5, 10 games)
    2. We can see performance trends (improving? declining?)
    3. We can analyze performance vs. specific teams

    Args:
        player_name: Full name of the player
        season: Format "YYYY-YY" (e.g., "2024-25")
        season_type: "Regular Season", "Playoffs", or "All Star"

    Returns:
        DataFrame with one row per game played
    """
    player_id = get_player_id(player_name)
    if player_id is None:
        return pd.DataFrame()

    try:
        # PlayerGameLog is an "endpoint" - it hits the NBA's API server
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type
        )

        # get_data_frames() returns a list of DataFrames
        # Most endpoints return just one, so we take [0]
        df = gamelog.get_data_frames()[0]

        # Add player name for clarity (the API only returns player_id)
        df["PLAYER_NAME"] = player_name

        return df

    except Exception as e:
        print(f"Error fetching stats for {player_name}: {e}")
        return pd.DataFrame()


def get_player_stats_multiple_seasons(
    player_name: str,
    seasons: list[str] = None,
    season_type: str = "Regular Season"
) -> pd.DataFrame:
    """
    Fetch game logs across multiple seasons for richer training data.

    WHY MULTIPLE SEASONS:
    More data = better ML models (usually). 3 seasons gives us:
    - ~250 games per player (if healthy)
    - Data from different team contexts
    - Long-term trends and consistency patterns

    Args:
        player_name: Full name of the player
        seasons: List of seasons like ["2024-25", "2023-24", "2022-23"]
                 Defaults to last 3 seasons if not provided
        season_type: "Regular Season" or "Playoffs"

    Returns:
        Combined DataFrame with all games from all seasons
    """
    if seasons is None:
        # Default to last 3 seasons
        seasons = ["2024-25", "2023-24", "2022-23"]

    all_games = []

    for season in seasons:
        print(f"Fetching {player_name}'s data for {season}...")
        df = get_player_stats(player_name, season, season_type)

        if not df.empty:
            df["SEASON"] = season  # Add column to track which season
            all_games.append(df)

        # IMPORTANT: Add delay between API calls to avoid rate limiting
        # The NBA API may temporarily block you if you make too many requests
        time.sleep(0.6)  # 600ms delay

    if not all_games:
        return pd.DataFrame()

    # pd.concat combines multiple DataFrames into one
    # ignore_index=True resets the row numbers (0, 1, 2, ...)
    return pd.concat(all_games, ignore_index=True)


def get_player_splits(
    player_name: str,
    season: str = "2024-25",
    season_type: str = "Regular Season"
) -> dict[str, pd.DataFrame]:
    """
    Fetch player performance SPLITS - breakdowns by different conditions.

    WHAT ARE SPLITS:
    Splits show how a player performs in different situations:
    - Home vs Away: Some players perform better at home (crowd support)
    - Wins vs Losses: Players may stat-pad in blowout wins
    - By Month: Early season rust vs. mid-season peak vs. late fatigue

    WHY SPLITS MATTER FOR PREDICTIONS:
    If a player averages 25 PPG overall but:
    - 28 PPG at home, 22 PPG away
    - 30 PPG vs bad defenses, 18 PPG vs elite defenses
    Then we need this context for accurate predictions!

    Returns:
        Dictionary with different split DataFrames:
        {
            "overall": overall season stats,
            "location": home vs away,
            "outcome": wins vs losses,
            "month": by month
        }
    """
    player_id = get_player_id(player_name)
    if player_id is None:
        return {}

    try:
        # This endpoint returns MANY different breakdowns at once
        # NOTE: Different endpoints use different parameter names for season_type
        # This one uses "season_type_playoffs" instead of "season_type_all_star"
        dashboard = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
            player_id=player_id,
            season=season,
            season_type_playoffs=season_type  # Different param name than other endpoints!
        )

        # The API returns multiple DataFrames for different split types
        # We extract the ones we care about
        frames = dashboard.get_data_frames()

        return {
            "overall": frames[0],      # Season totals
            "location": frames[1],     # Home vs Away
            "wins_losses": frames[2],  # Wins vs Losses
            "month": frames[3],        # By Month
            "pre_post_allstar": frames[4],  # Before/After All-Star break
        }

    except Exception as e:
        print(f"Error fetching splits for {player_name}: {e}")
        return {}


def get_all_players_season_stats(
    season: str = "2024-25",
    season_type: str = "Regular Season"
) -> pd.DataFrame:
    """
    Fetch season stats for ALL NBA players at once.

    WHY THIS IS USEFUL:
    1. Get data for every player in one API call (efficient)
    2. Compare players against each other
    3. Find league averages for normalization
    4. Identify which players to include in our model

    The API returns ~500 rows (all players who played that season).

    Returns:
        DataFrame with one row per player, columns include:
        - PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION
        - GP (games played), MIN (minutes per game)
        - PTS, AST, REB (per game averages)
        - Advanced: FG_PCT, FG3_PCT, FT_PCT, PLUS_MINUS
    """
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame"  # Get per-game averages, not totals
        )

        return stats.get_data_frames()[0]

    except Exception as e:
        print(f"Error fetching league player stats: {e}")
        return pd.DataFrame()


# =============================================================================
# TEAM DATA FUNCTIONS
# =============================================================================

def get_team_data(
    season: str = "2024-25",
    season_type: str = "Regular Season"
) -> pd.DataFrame:
    """
    Fetch team-level per-game stats for all NBA teams.

    WHAT THIS RETURNS:
    One row per team with stats like:
    - Offensive: PTS, FG_PCT, FG3_PCT, AST
    - Defensive: OPP_PTS (points allowed), DEF_RATING
    - Other: PACE (possessions per game), PLUS_MINUS

    WHY WE NEED TEAM DATA:
    Team context affects individual predictions:
    - Fast-paced teams = more possessions = more counting stats
    - Good teams = players play in more blowouts (less minutes in 4th quarter)

    Returns:
        DataFrame with one row per NBA team
    """
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
        )
        df = stats.get_data_frames()[0]

        # Filter out WNBA teams (the API sometimes includes them)
        # We get the list of NBA team IDs and only keep those
        nba_teams = teams.get_teams()
        nba_team_ids = {team["id"] for team in nba_teams}
        df = df[df["TEAM_ID"].isin(nba_team_ids)]

        return df

    except Exception as e:
        print(f"Error fetching team stats: {e}")
        return pd.DataFrame()


def get_team_defensive_stats(
    season: str = "2024-25",
    season_type: str = "Regular Season"
) -> pd.DataFrame:
    """
    Fetch DEFENSIVE stats for all teams.

    WHY DEFENSIVE STATS MATTER:
    This is CRITICAL for predictions. If a player is facing:
    - Elite defense (top 5): Expect LOWER stats
    - Poor defense (bottom 5): Expect HIGHER stats

    Key defensive metrics:
    - DEF_RATING: Points allowed per 100 possessions (lower = better defense)
    - OPP_PTS: Average points allowed per game
    - OPP_FG_PCT: Opponent field goal percentage

    Returns:
        DataFrame with defensive stats for each team
    """
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Opponent"  # KEY: Gets opponent/defensive stats
        )
        df = stats.get_data_frames()[0]

        # Filter to NBA teams only
        nba_teams = teams.get_teams()
        nba_team_ids = {team["id"] for team in nba_teams}
        df = df[df["TEAM_ID"].isin(nba_team_ids)]

        return df

    except Exception as e:
        print(f"Error fetching defensive stats: {e}")
        return pd.DataFrame()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_active_players(min_games: int = 10, season: str = "2024-25") -> list[str]:
    """
    Get a list of currently active players who have played enough games.

    WHY FILTER BY GAMES PLAYED:
    We don't want to include:
    - Injured players who played 2 games
    - End-of-bench players with tiny sample sizes
    - Two-way (G-League) players with limited NBA time

    For ML, we need players with enough data to learn patterns.

    Args:
        min_games: Minimum games played to be included (default 10)
        season: Which season to check

    Returns:
        List of player names who meet the criteria
    """
    df = get_all_players_season_stats(season)

    if df.empty:
        return []

    # Filter by games played
    active = df[df["GP"] >= min_games]

    return active["PLAYER_NAME"].tolist()


def get_player_info(player_name: str) -> dict | None:
    """
    Get static info about a player (position, height, etc.)

    WHY THIS MATTERS:
    Position affects what stats to expect:
    - Centers: More rebounds, fewer assists
    - Point Guards: More assists, fewer rebounds
    - Height: Taller players rebound more

    Returns:
        Dictionary with player info or None if not found
    """
    player_dict = players.find_players_by_full_name(player_name)

    if not player_dict:
        return None

    return player_dict[0]
