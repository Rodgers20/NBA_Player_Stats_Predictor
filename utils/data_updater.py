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
from pathlib import Path
import pandas as pd


# Persist last download date to disk so it survives app restarts
_STAMP_FILE = Path(__file__).parent.parent / "data" / ".last_kaggle_download"

def _read_stamp() -> str | None:
    try:
        return _STAMP_FILE.read_text().strip() if _STAMP_FILE.exists() else None
    except Exception:
        return None

def _write_stamp(date_str: str) -> None:
    try:
        _STAMP_FILE.write_text(date_str)
    except Exception:
        pass

# Track state
_last_update_date = None
_last_download_date = _read_stamp()  # Persist across restarts
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
    global _last_update_date, _last_download_date, _is_updating

    from utils.kaggle_loader import download_dataset, load_player_game_logs

    # Check if already updating
    with _update_lock:
        if _is_updating:
            print("[DataUpdater] Update already in progress, skipping")
            return False
        _is_updating = True

    try:
        today = datetime.now().date()
        print(f"[DataUpdater] Starting update at {datetime.now()}")

        # 1. Update rosters (handle trades)
        update_rosters()

        # 2. Refresh Kaggle dataset (force-download once per day, cache otherwise)
        today_str = str(today)
        if _last_download_date != today_str:
            print("[DataUpdater] Daily refresh: downloading latest Kaggle data...")
            try:
                download_dataset(force=True)
                _last_download_date = today_str
                _write_stamp(today_str)
            except Exception as dl_err:
                print(f"[DataUpdater] Force download failed ({dl_err}), using cached data")
                download_dataset(force=False)
        else:
            print("[DataUpdater] Using cached Kaggle data (already downloaded today)")
            download_dataset(force=False)

        # 3. Load current season from Kaggle
        fresh_df = load_player_game_logs(num_seasons=1)

        if fresh_df.empty:
            print("[DataUpdater] No data from Kaggle")
            return False

        # 3b. Check if Kaggle data is stale — if max date < yesterday, top-up via NBA API
        try:
            fresh_df["_tmp_date"] = pd.to_datetime(
                fresh_df["GAME_DATE"], format="%b %d, %Y", errors="coerce"
            )
            kaggle_max = fresh_df["_tmp_date"].max()
            fresh_df.drop(columns=["_tmp_date"], inplace=True)
            yesterday = (today - timedelta(days=1))
            if pd.notnull(kaggle_max) and kaggle_max.date() < yesterday:
                gap_from = kaggle_max.strftime("%Y-%m-%d")
                print(f"[DataUpdater] Kaggle data ends {kaggle_max.date()}, fetching gap from NBA API ...")
                api_df = _fetch_nba_api_games(gap_from)
                if not api_df.empty:
                    fresh_df = pd.concat([fresh_df, api_df], ignore_index=True).drop_duplicates(
                        subset=["PLAYER_NAME", "GAME_DATE", "MATCHUP"], keep="last"
                    )
                    print(f"[DataUpdater] After NBA API top-up: {len(fresh_df)} rows total")
        except Exception as _gap_err:
            print(f"[DataUpdater] Gap-fill check failed (non-critical): {_gap_err}")

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


def _fetch_nba_api_games(from_date: str) -> pd.DataFrame:
    """
    Fetch player game logs directly from NBA Stats API for dates after from_date.
    Used when Kaggle dataset is stale.

    Args:
        from_date: "YYYY-MM-DD" — fetch games after this date.

    Returns:
        DataFrame with same schema as Kaggle pipeline (PLAYER_NAME, GAME_DATE, etc.)
        Empty DataFrame on failure.
    """
    try:
        from nba_api.stats.endpoints import LeagueGameLog
        from datetime import datetime as _dt

        from_dt = _dt.strptime(from_date, "%Y-%m-%d")
        date_from_str = from_dt.strftime("%m/%d/%Y")   # NBA API format: MM/DD/YYYY

        print(f"[DataUpdater] Fetching NBA Stats API games from {date_from_str} ...")
        lg = LeagueGameLog(
            season="2025-26",
            player_or_team_abbreviation="P",   # player-level stats
            date_from_nullable=date_from_str,
            timeout=30,
        )
        df = lg.get_data_frames()[0]
        if df.empty:
            print("[DataUpdater] NBA Stats API returned no games")
            return pd.DataFrame()

        # Rename columns to match pipeline schema
        rename_map = {
            "TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
            "PLAYER_NAME": "PLAYER_NAME",
            "GAME_DATE": "GAME_DATE",   # format: "2026-03-24"
            "MATCHUP": "MATCHUP",
            "WL": "WL",
            "MIN": "MIN",
            "PTS": "PTS",
            "REB": "REB",
            "AST": "AST",
            "FG3M": "FG3M",
            "FGM": "FGM",
            "FGA": "FGA",
            "FG3A": "FG3A",
            "FTM": "FTM",
            "FTA": "FTA",
            "STL": "STL",
            "BLK": "BLK",
            "TOV": "TOV",
        }
        # Keep only columns we care about
        available = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df[list(available.keys())].rename(columns=available)

        # Convert GAME_DATE from "2026-03-24" to "Mar 24, 2026"
        def _reformat_date(d):
            try:
                return _dt.strptime(str(d), "%Y-%m-%d").strftime("%b %d, %Y")
            except Exception:
                return d
        df["GAME_DATE"] = df["GAME_DATE"].apply(_reformat_date)

        print(f"[DataUpdater] NBA Stats API: {len(df)} player-game rows fetched")
        return df

    except Exception as e:
        import traceback
        print(f"[DataUpdater] NBA Stats API fallback failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def get_last_update_time():
    """Get the timestamp of the last successful update."""
    return _last_update_date


def is_update_in_progress():
    """Check if an update is currently running."""
    return _is_updating
