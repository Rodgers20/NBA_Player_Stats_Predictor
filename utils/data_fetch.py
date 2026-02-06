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
    scoreboardv2,  # For today's games
)
from nba_api.stats.static import players, teams
from datetime import datetime


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


# =============================================================================
# POSITION AND MATCHUP FUNCTIONS
# =============================================================================

def get_all_players_with_positions(season: str = "2024-25") -> pd.DataFrame:
    """
    Get all active players with their positions.

    WHY POSITION MATTERS:
    - Centers average more rebounds than guards
    - Point guards average more assists than forwards
    - Position affects baseline stat expectations

    Uses LeagueDashPlayerStats which includes position info.

    Returns:
        DataFrame with PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, POSITION
    """
    from nba_api.stats.endpoints import commonplayerinfo

    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame"
        )
        df = stats.get_data_frames()[0]

        # Keep relevant columns
        result = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]].copy()

        # Infer position from stats patterns
        # Guards: high AST, low REB
        # Forwards: balanced stats
        # Centers: high REB, low AST
        def infer_position(row):
            ast = df[df["PLAYER_ID"] == row["PLAYER_ID"]]["AST"].values
            reb = df[df["PLAYER_ID"] == row["PLAYER_ID"]]["REB"].values

            if len(ast) == 0 or len(reb) == 0:
                return "F"

            ast_val = ast[0]
            reb_val = reb[0]

            # Simple heuristics based on typical position stats
            if ast_val >= 4.0 and reb_val < 5.0:
                return "G"  # Guard - high assists, lower rebounds
            elif reb_val >= 7.0:
                return "C"  # Center - high rebounds
            else:
                return "F"  # Forward - default

        result["POSITION"] = result.apply(infer_position, axis=1)

        return result

    except Exception as e:
        print(f"Error fetching player positions: {e}")
        return pd.DataFrame()


def extract_opponent_from_matchup(matchup: str) -> str:
    """
    Extract opponent team abbreviation from matchup string.

    Examples:
        "LAL vs. GSW" -> "GSW"
        "LAL @ BOS" -> "BOS"
    """
    if not isinstance(matchup, str):
        return ""

    if "@" in matchup:
        return matchup.split("@")[-1].strip()[:3]
    elif "vs." in matchup:
        return matchup.split("vs.")[-1].strip()[:3]
    return ""


def calculate_defense_vs_position(
    game_logs_df: pd.DataFrame,
    player_positions_df: pd.DataFrame,
    season: str = None
) -> pd.DataFrame:
    """
    Calculate team defensive stats vs each position (G, F, C).

    METHODOLOGY:
    1. For each game, identify the opponent team
    2. Group by opponent team and player position
    3. Calculate average PTS, REB, AST allowed per position
    4. Rank teams 1-30 for each position/stat combo

    WHY THIS MATTERS:
    Some teams are great at defending guards but weak vs. centers.
    This position-specific data helps make better predictions.

    Args:
        game_logs_df: Player game logs with MATCHUP, PTS, AST, REB columns
        player_positions_df: DataFrame with PLAYER_ID and POSITION columns
        season: Optional season filter

    Returns:
        DataFrame with columns:
        TEAM_ABBREVIATION, POSITION, SEASON,
        AVG_PTS_ALLOWED, AVG_REB_ALLOWED, AVG_AST_ALLOWED,
        PTS_RANK, REB_RANK, AST_RANK
    """
    df = game_logs_df.copy()

    # Merge position info
    df = df.merge(
        player_positions_df[["PLAYER_ID", "POSITION"]],
        left_on="Player_ID",
        right_on="PLAYER_ID",
        how="left"
    )
    df["POSITION"] = df["POSITION"].fillna("F")

    # Extract opponent from matchup
    df["OPPONENT"] = df["MATCHUP"].apply(extract_opponent_from_matchup)

    # Filter by season if specified
    if season:
        df = df[df["SEASON"] == season]

    # Group by opponent, position, season
    grouped = df.groupby(["OPPONENT", "POSITION", "SEASON"]).agg({
        "PTS": "mean",
        "REB": "mean",
        "AST": "mean",
        "FG3M": "mean"
    }).reset_index()

    grouped.columns = [
        "TEAM_ABBREVIATION", "POSITION", "SEASON",
        "AVG_PTS_ALLOWED", "AVG_REB_ALLOWED", "AVG_AST_ALLOWED", "AVG_3PM_ALLOWED"
    ]

    # Add rankings within each position and season (higher = worse defense)
    for stat in ["PTS", "REB", "AST", "3PM"]:
        col = f"AVG_{stat}_ALLOWED"
        rank_col = f"{stat}_RANK"
        grouped[rank_col] = grouped.groupby(["POSITION", "SEASON"])[col].rank(
            ascending=False, method="min"
        ).astype(int)

    return grouped


# =============================================================================
# LIVE GAME FUNCTIONS
# =============================================================================

def get_todays_games() -> pd.DataFrame:
    """
    Fetch today's scheduled NBA games.

    WHY THIS IS NEEDED:
    For the "Best Props" feature, we need to know which games
    are being played today and which players will be active.

    Returns:
        DataFrame with columns:
        GAME_ID, GAME_DATE, HOME_TEAM_ID, HOME_TEAM_ABBREVIATION,
        VISITOR_TEAM_ID, VISITOR_TEAM_ABBREVIATION, GAME_STATUS
    """
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        scoreboard = scoreboardv2.ScoreboardV2(
            game_date=today,
            league_id="00",
            day_offset=0
        )

        # Get the game header which contains basic game info
        games_df = scoreboard.get_data_frames()[0]  # GameHeader

        if games_df.empty:
            return pd.DataFrame()

        # Select relevant columns
        columns_to_keep = [
            "GAME_ID", "GAME_DATE_EST", "HOME_TEAM_ID",
            "VISITOR_TEAM_ID", "GAME_STATUS_TEXT"
        ]
        available_cols = [c for c in columns_to_keep if c in games_df.columns]
        result = games_df[available_cols].copy()

        # Add team abbreviations
        all_teams = {t["id"]: t["abbreviation"] for t in teams.get_teams()}

        if "HOME_TEAM_ID" in result.columns:
            result["HOME_TEAM"] = result["HOME_TEAM_ID"].map(all_teams)
        if "VISITOR_TEAM_ID" in result.columns:
            result["AWAY_TEAM"] = result["VISITOR_TEAM_ID"].map(all_teams)

        return result

    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return pd.DataFrame()


def get_recent_games_for_player(
    player_name: str,
    since_date: str = None
) -> pd.DataFrame:
    """
    Fetch recent games for a player, optionally since a specific date.

    WHY THIS IS NEEDED:
    For real-time updates, we don't want to re-download entire seasons.
    This fetches only new games since the last update.

    Args:
        player_name: Full player name
        since_date: Only return games after this date (format: "YYYY-MM-DD")

    Returns:
        DataFrame with game logs since the specified date
    """
    # Fetch current season data
    df = get_player_stats(player_name, season="2024-25")

    if df.empty:
        return df

    if since_date:
        # Parse game dates and filter
        df["_date"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
        since = pd.to_datetime(since_date)
        df = df[df["_date"] > since]
        df = df.drop(columns=["_date"])

    return df


def get_teams_playing_today() -> list[str]:
    """
    Get list of team abbreviations playing today.

    Returns:
        List of team abbreviations (e.g., ["LAL", "GSW", "BOS", "MIA"])
    """
    games = get_todays_games()

    if games.empty:
        return []

    teams_today = []
    if "HOME_TEAM" in games.columns:
        teams_today.extend(games["HOME_TEAM"].dropna().tolist())
    if "AWAY_TEAM" in games.columns:
        teams_today.extend(games["AWAY_TEAM"].dropna().tolist())

    return list(set(teams_today))


def get_next_opponent_for_team(team_abbrev: str, max_days: int = 7) -> tuple[str, str]:
    """
    Get the next opponent for a team by checking upcoming days.

    Args:
        team_abbrev: Team abbreviation (e.g., "BOS", "LAL")
        max_days: Maximum number of days to look ahead

    Returns:
        Tuple of (opponent_abbreviation, game_date) or ("", "") if not found
    """
    from datetime import datetime, timedelta

    all_teams = {t["id"]: t["abbreviation"] for t in teams.get_teams()}

    for day_offset in range(max_days):
        check_date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        try:
            scoreboard = scoreboardv2.ScoreboardV2(
                game_date=check_date,
                league_id="00",
                day_offset=0
            )
            games_df = scoreboard.get_data_frames()[0]

            if games_df.empty:
                continue

            # Add team abbreviations
            if "HOME_TEAM_ID" in games_df.columns:
                games_df["HOME_TEAM"] = games_df["HOME_TEAM_ID"].map(all_teams)
            if "VISITOR_TEAM_ID" in games_df.columns:
                games_df["AWAY_TEAM"] = games_df["VISITOR_TEAM_ID"].map(all_teams)

            # Check if our team is playing
            for _, game in games_df.iterrows():
                home = game.get("HOME_TEAM", "")
                away = game.get("AWAY_TEAM", "")
                if team_abbrev == home:
                    return away, check_date
                elif team_abbrev == away:
                    return home, check_date

            time.sleep(0.3)  # Rate limiting

        except Exception as e:
            print(f"Error checking date {check_date}: {e}")
            continue

    return "", ""
