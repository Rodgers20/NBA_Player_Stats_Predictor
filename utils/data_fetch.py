# utils/data_fetch.py

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, playergamelog
from nba_api.stats.static import players
from nba_api.stats.library.parameters import SeasonAll


from nba_api.stats.static import teams


def get_player_stats(
    player_name: str, season="2025-26", season_type="Regular Season"
) -> pd.DataFrame:
    """
    Fetch all game logs for a given player in a specified season.

    Args:
        player_name (str): Full name of the player (e.g. "LeBron James").
        season (str): Season in format "YYYY-YY" (e.g. "2025-26").

    Returns:
        pd.DataFrame: DataFrame with the player's game-by-game stats.
    """
    # Find player ID
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        print(f"Player '{player_name}' not found.")
        return pd.DataFrame()

    player_id = player_dict[0]["id"]

    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star=season_type
        )
        df = gamelog.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching stats for {player_name}: {e}")
        return pd.DataFrame()


def get_team_data(season="2025-26", season_type="Regular Season") -> pd.DataFrame:
    """
    Fetch team-level per-game stats for a specific NBA season.

    Returns:
        pd.DataFrame: DataFrame of team stats per game.
    """
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
        )
        df = stats.get_data_frames()[0]

        # Filter out WNBA teams using known NBA team IDs
        nba_teams = teams.get_teams()
        nba_team_ids = {team["id"] for team in nba_teams}
        df = df[df["TEAM_ID"].isin(nba_team_ids)]

        return df
    except Exception as e:
        print(f"Error fetching team stats: {e}")
        return pd.DataFrame()
