"""
Background data updater for NBA Player Stats Predictor.
Periodically refreshes data from Kaggle without restarting the app.

This module handles:
- Downloading latest Kaggle dataset (updated daily)
- Diffing against existing data to find new games
- Thread-safe merges into global state
"""
import os
import threading
from datetime import datetime, timedelta
import pandas as pd


# Track state
_last_update_date = None
_update_lock = threading.Lock()
_is_updating = False


def get_players_who_played_today(roster_df, teams_today):
    """
    Filter to only active players on teams that played today.
    Uses the latest ROSTER (not game logs) to ensure traded players are found.

    Args:
        roster_df: DataFrame with PLAYER_NAME and TEAM_ABBREVIATION columns
        teams_today: List of team abbreviations that played today

    Returns:
        List of player names
    """
    if not teams_today:
        return []

    if roster_df.empty:
        print("[DataUpdater] Roster is empty, cannot determine active players")
        return []

    if "TEAM_ABBREVIATION" not in roster_df.columns:
        print("[DataUpdater] No TEAM_ABBREVIATION column in roster")
        return []

    active_players = roster_df[
        roster_df["TEAM_ABBREVIATION"].isin(teams_today)
    ]["PLAYER_NAME"].unique().tolist()

    return active_players


def update_rosters():
    """
    Fetch latest player positions from Kaggle dataset.
    Handles traded players by reflecting their current team.
    """
    from utils.kaggle_loader import load_player_positions

    print("[DataUpdater] Updating roster from Kaggle data...")
    try:
        positions_df = load_player_positions(num_seasons=1)

        if not positions_df.empty:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            positions_path = os.path.join(data_dir, "player_positions.csv")
            positions_df.to_csv(positions_path, index=False)
            print(f"[DataUpdater] Roster updated: {len(positions_df)} players.")
            return positions_df
        else:
            print("[DataUpdater] No positions loaded, using existing.")
            return pd.DataFrame()

    except Exception as e:
        print(f"[DataUpdater] Error updating rosters: {e}")
        return pd.DataFrame()


def update_game_data(get_df_func, merge_func):
    """
    Main update function - called by scheduler.

    Downloads latest Kaggle data and diffs against existing to find new games.

    Args:
        get_df_func: Function that returns current global DataFrame
        merge_func: Function to merge new games into global state

    Returns:
        True if update succeeded, False otherwise
    """
    global _last_update_date, _is_updating

    from utils.kaggle_loader import download_dataset, load_player_game_logs

    # Check if already updating
    with _update_lock:
        if _is_updating:
            print("[DataUpdater] Update already in progress, skipping")
            return False
        _is_updating = True

    try:
        print(f"[DataUpdater] Starting update at {datetime.now()}")

        # 1. Update rosters (handle trades)
        update_rosters()

        # 2. Refresh Kaggle dataset
        print("[DataUpdater] Downloading latest Kaggle data...")
        download_dataset(force=True)

        # 3. Load current season from Kaggle
        fresh_df = load_player_game_logs(num_seasons=1)

        if fresh_df.empty:
            print("[DataUpdater] No data from Kaggle")
            return False

        # 4. Diff against existing data
        current_df = get_df_func()

        if not current_df.empty:
            existing_keys = set(zip(
                current_df["PLAYER_NAME"],
                current_df["GAME_DATE"],
                current_df["MATCHUP"]
            ))
            new_mask = fresh_df.apply(
                lambda r: (r["PLAYER_NAME"], r["GAME_DATE"], r["MATCHUP"])
                not in existing_keys,
                axis=1
            )
            new_games = fresh_df[new_mask]
        else:
            new_games = fresh_df

        # 5. Merge new games
        if not new_games.empty:
            print(f"[DataUpdater] Found {len(new_games)} new game records")
            merge_func(new_games)
            _last_update_date = datetime.now()
            print(f"[DataUpdater] Update completed at {_last_update_date}")
            return True
        else:
            print("[DataUpdater] Data is up to date, no new games")
            _last_update_date = datetime.now()
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
