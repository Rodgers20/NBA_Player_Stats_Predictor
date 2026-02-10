"""
Background data updater for NBA Player Stats Predictor.
Periodically fetches new game data without restarting the app.

This module handles:
- Fetching recent games for players who played today
- Rate limiting to avoid NBA API bans
- Thread-safe updates to global data
"""
import time
import threading
from datetime import datetime, timedelta
import pandas as pd


# Track state
_last_update_date = None
_update_lock = threading.Lock()
_is_updating = False


def get_players_who_played_today(all_players_df, teams_today):
    """
    Filter to only players on teams that played today.

    Args:
        all_players_df: DataFrame with PLAYER_NAME and TEAM_ABBREVIATION columns
        teams_today: List of team abbreviations that played today

    Returns:
        List of player names
    """
    if not teams_today:
        return []

    if "TEAM_ABBREVIATION" not in all_players_df.columns:
        # Fallback: try to extract from MATCHUP column
        print("[DataUpdater] No TEAM_ABBREVIATION column, returning all players")
        return all_players_df["PLAYER_NAME"].unique().tolist()[:50]  # Limit to 50

    return all_players_df[
        all_players_df["TEAM_ABBREVIATION"].isin(teams_today)
    ]["PLAYER_NAME"].unique().tolist()


def fetch_games_batch(players, since_date, season, delay=2.0):
    """
    Fetch recent games for a batch of players with rate limiting.

    Args:
        players: List of player names
        since_date: Only fetch games after this datetime
        season: NBA season string (e.g., "2025-26")
        delay: Seconds to wait between API calls

    Returns:
        DataFrame with all new games
    """
    # Import here to avoid circular imports
    from utils.data_fetch import get_player_stats, get_current_nba_season

    all_games = []
    for player in players:
        try:
            df = get_player_stats(player, season=season)
            if not df.empty and since_date:
                df["_temp_date"] = pd.to_datetime(
                    df["GAME_DATE"],
                    format="%b %d, %Y",
                    errors="coerce"
                )
                df = df[df["_temp_date"] > since_date]
                df = df.drop(columns=["_temp_date"])

            if not df.empty:
                df["PLAYER_NAME"] = player
                df["SEASON"] = season
                all_games.append(df)
                print(f"[DataUpdater] Fetched {len(df)} games for {player}")

            time.sleep(delay)  # Rate limit

        except Exception as e:
            print(f"[DataUpdater] Error fetching {player}: {e}")
            time.sleep(delay * 2)  # Extra delay on error

    if all_games:
        return pd.concat(all_games, ignore_index=True)
    return pd.DataFrame()


def update_game_data(get_df_func, merge_func):
    """
    Main update function - called by scheduler every 30 minutes.

    Args:
        get_df_func: Function that returns current global DataFrame
        merge_func: Function to merge new games into global state

    Returns:
        True if update succeeded, False otherwise
    """
    global _last_update_date, _is_updating

    # Import here to avoid circular imports
    from utils.data_fetch import get_teams_playing_today, get_current_nba_season

    # Check if already updating
    with _update_lock:
        if _is_updating:
            print("[DataUpdater] Update already in progress, skipping")
            return False
        _is_updating = True

    try:
        print(f"[DataUpdater] Starting update at {datetime.now()}")

        # Get teams that played today
        teams_today = get_teams_playing_today()
        if not teams_today:
            print("[DataUpdater] No games today, skipping update")
            return False

        print(f"[DataUpdater] Teams playing today: {teams_today}")

        # Calculate since_date (yesterday if first run, else last update)
        since_date = _last_update_date or (datetime.now() - timedelta(days=1))

        # Get current DataFrame and extract players from teams that played
        current_df = get_df_func()
        players = get_players_who_played_today(current_df, teams_today)

        if not players:
            print("[DataUpdater] No players found for today's teams")
            return False

        print(f"[DataUpdater] Updating {len(players)} players from {len(teams_today)} teams")

        # Get current season
        season = get_current_nba_season()

        # Fetch in batches of 20 with 30s pause between batches
        BATCH_SIZE = 20
        BATCH_DELAY = 30  # seconds between batches
        API_DELAY = 2.0   # seconds between individual API calls

        all_new_games = []
        total_batches = (len(players) - 1) // BATCH_SIZE + 1

        for i in range(0, len(players), BATCH_SIZE):
            batch = players[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            print(f"[DataUpdater] Processing batch {batch_num}/{total_batches}")

            new_df = fetch_games_batch(batch, since_date, season, delay=API_DELAY)
            if not new_df.empty:
                all_new_games.append(new_df)

            # Pause between batches (but not after last batch)
            if i + BATCH_SIZE < len(players):
                print(f"[DataUpdater] Pausing {BATCH_DELAY}s before next batch...")
                time.sleep(BATCH_DELAY)

        if all_new_games:
            combined = pd.concat(all_new_games, ignore_index=True)
            print(f"[DataUpdater] Found {len(combined)} new game records")

            # Merge into global state via callback
            merge_func(combined)

            _last_update_date = datetime.now()
            print(f"[DataUpdater] Update completed successfully at {_last_update_date}")
            return True
        else:
            print("[DataUpdater] No new games found")
            _last_update_date = datetime.now()  # Still update timestamp
            return False

    except Exception as e:
        print(f"[DataUpdater] Error during update: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        with _update_lock:
            _is_updating = False


def get_last_update_time():
    """Get the timestamp of the last successful update."""
    return _last_update_date


def is_update_in_progress():
    """Check if an update is currently running."""
    return _is_updating
