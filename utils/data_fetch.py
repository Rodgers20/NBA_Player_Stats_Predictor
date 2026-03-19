# utils/data_fetch.py
"""
NBA Data Fetching Module (Live Functions Only)
===============================================
This module handles live/real-time NBA API calls.

Historical data fetching (player stats, team stats, defensive stats, positions)
has been moved to utils/kaggle_loader.py which reads from the Kaggle dataset.

This module retains:
- Live game schedule lookups (today's games, next opponent)
- Utility functions (player/team ID lookup, matchup parsing)
- Defense vs position calculation (pure computation, no API)
"""

import time
import random
import pandas as pd
from nba_api.stats.static import players, teams
from datetime import datetime, timedelta

# Custom headers to avoid NBA API blocks
CUSTOM_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}

# Proxy configuration (set to None to disable)
PROXY = None


def get_proxy_dict():
    """Get proxy dictionary for requests if PROXY is configured."""
    if PROXY:
        return {"http": PROXY, "https": PROXY}
    return None


def get_current_nba_season():
    """
    Get the current NBA season string (e.g., '2025-26').

    NBA season runs Oct-June:
    - Oct-Dec 2025 = '2025-26' season
    - Jan-Sep 2026 = still '2025-26' season
    """
    today = datetime.now()
    year = today.year
    month = today.month

    if month >= 10:  # October or later
        start_year = year
    else:  # January - September
        start_year = year - 1

    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def api_call_with_retry(api_func, max_retries=3, base_delay=5):
    """
    Wrapper to retry NBA API calls with exponential backoff.

    Args:
        api_func: A callable that makes the API call
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)

    Returns:
        The result of api_func() or raises the last exception
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"  Retry {attempt}/{max_retries} after {delay:.1f}s delay...")
                time.sleep(delay)

            return api_func()

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            if "timeout" in error_msg or "connection" in error_msg:
                print(f"  Connection issue: {e}")
                continue
            else:
                raise e

    raise last_exception


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_player_id(player_name: str) -> int | None:
    """
    Convert a player's name to their NBA ID.

    Args:
        player_name: Full name like "LeBron James"

    Returns:
        The player's NBA ID (int) or None if not found
    """
    player_dict = players.find_players_by_full_name(player_name)

    if not player_dict:
        print(f"Player '{player_name}' not found.")
        return None

    return player_dict[0]["id"]


def get_team_id(team_name: str) -> int | None:
    """
    Convert a team's name to their NBA ID.

    Args:
        team_name: Can be full name, city, or abbreviation

    Returns:
        The team's NBA ID or None if not found
    """
    all_teams = teams.get_teams()

    for team in all_teams:
        if (team_name.lower() in team["full_name"].lower() or
            team_name.lower() == team["abbreviation"].lower()):
            return team["id"]

    print(f"Team '{team_name}' not found.")
    return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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

# Simple cache for today's games to avoid repeated API calls
_todays_games_cache = {"data": None, "timestamp": None}
_CACHE_TTL_SECONDS = 300  # Cache for 5 minutes

# ESPN uses shortened abbreviations that differ from NBA/Kaggle standard.
# Map ESPN → NBA standard so player lookups match PLAYER_POSITIONS (Kaggle data).
# NBA standard abbreviations: SAS, GSW, NYK, NOP, WAS, PHX, etc.
_ESPN_ABBREV_MAP = {
    "SA":   "SAS",   # San Antonio Spurs  (ESPN: SA  → NBA: SAS)
    "NO":   "NOP",   # New Orleans Pelicans (ESPN: NO  → NBA: NOP)
    "NY":   "NYK",   # New York Knicks     (ESPN: NY  → NBA: NYK)
    "GS":   "GSW",   # Golden State Warriors (ESPN: GS → NBA: GSW)
    "WSH":  "WAS",   # Washington Wizards  (ESPN: WSH → NBA: WAS)
    "PHX":  "PHX",   # Phoenix Suns        (consistent)
    "UTAH": "UTA",   # Utah Jazz           (ESPN sometimes returns 4-letter)
    "NJ":   "BKN",   # Brooklyn Nets       (legacy ESPN code)
    "BKN":  "BKN",   # Brooklyn Nets       (modern, explicit)
    "CHA":  "CHA",   # Charlotte Hornets   (explicit)
    "OKC":  "OKC",   # Oklahoma City       (explicit)
}


def _normalize_espn_abbrev(abbrev: str) -> str:
    return _ESPN_ABBREV_MAP.get(abbrev, abbrev)


def _get_games_from_espn(date_str: str) -> pd.DataFrame:
    """Fetch games from ESPN's public scoreboard API for the given date (no auth).

    Args:
        date_str: Date in "YYYY-MM-DD" format. ESPN accepts ?dates=YYYYMMDD.
    """
    import requests as _requests
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        # ESPN uses YYYYMMDD format; omit param for today (defaults to current date)
        espn_date = date_str.replace("-", "")
        resp = _requests.get(url, headers=CUSTOM_HEADERS, params={"dates": espn_date}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for event in data.get("events", []):
            comp = event["competitions"][0]
            home_team, away_team = "", ""
            for competitor in comp["competitors"]:
                abbrev = _normalize_espn_abbrev(competitor["team"]["abbreviation"])
                if competitor["homeAway"] == "home":
                    home_team = abbrev
                else:
                    away_team = abbrev

            raw_date = event.get("date", "")  # "2026-03-07T23:30:00Z"
            game_date = raw_date[:10]
            rows.append({
                "GAME_ID": event["id"],
                "GAME_DATE_EST": game_date,
                "GAME_TIME": raw_date,  # full ISO timestamp for tip-off display
                "HOME_TEAM": home_team,
                "AWAY_TEAM": away_team,
                "GAME_STATUS_TEXT": event["status"]["type"]["description"],
                "HOME_TEAM_ID": 0,
                "VISITOR_TEAM_ID": 0,
            })

        if rows:
            print(f"[Games] ESPN returned {len(rows)} games for {date_str}")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[Games] ESPN fetch failed for {date_str}: {e}")
        return pd.DataFrame()


def get_todays_games() -> pd.DataFrame:
    """
    Fetch today's scheduled NBA games with caching.
    Uses ESPN (fast, reliable, no auth). NBA API is dead and always times out.

    Returns:
        DataFrame with columns:
        GAME_ID, GAME_DATE_EST, HOME_TEAM, AWAY_TEAM, GAME_STATUS_TEXT
    """
    global _todays_games_cache

    now = datetime.now()
    if _todays_games_cache["data"] is not None and _todays_games_cache["timestamp"]:
        age = (now - _todays_games_cache["timestamp"]).total_seconds()
        if age < _CACHE_TTL_SECONDS:
            return _todays_games_cache["data"]

    today = now.strftime("%Y-%m-%d")
    result = _get_games_from_espn(today)

    if not result.empty:
        print(f"[Games] Found {len(result)} games for {today}")

    _todays_games_cache["data"] = result
    _todays_games_cache["timestamp"] = datetime.now()
    return result


# Cache for upcoming games (separate from today's cache since it may target tomorrow)
_upcoming_games_cache = {"data": None, "target_date": None, "timestamp": None}


def get_upcoming_games() -> tuple[pd.DataFrame, str]:
    """
    Get games that haven't finished yet, with fallback to tomorrow's slate.

    Logic:
    1. Fetch today's games from ESPN
    2. Filter out games with status "Final" (already played)
    3. If no unplayed games today, fetch tomorrow's full schedule
    4. Cache for 5 minutes

    Returns:
        (games_df, target_date_str) — target_date_str is "YYYY-MM-DD"
    """
    global _upcoming_games_cache

    now = datetime.now()
    if (
        _upcoming_games_cache["data"] is not None
        and _upcoming_games_cache["timestamp"]
        and (now - _upcoming_games_cache["timestamp"]).total_seconds() < _CACHE_TTL_SECONDS
    ):
        return _upcoming_games_cache["data"], _upcoming_games_cache["target_date"]

    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    # ESPN is the primary (and only) source — fast, no auth, always works
    today_games = _get_games_from_espn(today)

    # Filter to only unplayed / in-progress games
    if not today_games.empty and "GAME_STATUS_TEXT" in today_games.columns:
        upcoming_today = today_games[
            ~today_games["GAME_STATUS_TEXT"].str.lower().str.contains("final", na=False)
        ]
        if not upcoming_today.empty:
            print(f"[Games] {len(upcoming_today)} upcoming game(s) today ({today})")
            _upcoming_games_cache = {"data": upcoming_today, "target_date": today, "timestamp": now}
            return upcoming_today, today

    # All today's games are done (or no games today) — use tomorrow's slate
    print(f"[Games] No upcoming games today, fetching tomorrow ({tomorrow})...")
    tomorrow_games = _get_games_from_espn(tomorrow)

    target = tomorrow if not tomorrow_games.empty else today
    result = tomorrow_games if not tomorrow_games.empty else pd.DataFrame()
    _upcoming_games_cache = {"data": result, "target_date": target, "timestamp": now}
    return result, target


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


def get_teams_playing_between(start_date: str, end_date: str) -> list[str]:
    """
    Get list of team abbreviations that played between start_date and end_date.

    Args:
        start_date: Format "YYYY-MM-DD"
        end_date: Format "YYYY-MM-DD"

    Returns:
        List of team abbreviations
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    curr = start

    all_teams_set = set()

    while curr <= end:
        date_str = curr.strftime("%Y-%m-%d")
        # ESPN only — NBA API (stats.nba.com) consistently times out
        games_df = _get_games_from_espn(date_str)
        if not games_df.empty:
            if "HOME_TEAM" in games_df.columns:
                all_teams_set.update(games_df["HOME_TEAM"].dropna().tolist())
            if "AWAY_TEAM" in games_df.columns:
                all_teams_set.update(games_df["AWAY_TEAM"].dropna().tolist())
        curr += timedelta(days=1)

    return list(all_teams_set)


def get_next_opponent_for_team(team_abbrev: str, max_days: int = 7) -> tuple[str, str]:
    """
    Get the next opponent for a team, looking ahead up to max_days.

    Uses ESPN as primary source (reliable, no auth) with NBA API as fallback.

    Args:
        team_abbrev: Team abbreviation (e.g., "BOS", "LAL")
        max_days: Maximum number of days to look ahead

    Returns:
        Tuple of (opponent_abbreviation, game_date) or ("", "") if not found
    """
    for day_offset in range(max_days):
        check_date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        # ESPN only — NBA API (stats.nba.com) consistently times out
        games_df = _get_games_from_espn(check_date)

        if games_df.empty:
            continue

        for _, game in games_df.iterrows():
            home = game.get("HOME_TEAM", "")
            away = game.get("AWAY_TEAM", "")
            if team_abbrev == home:
                return away, check_date
            elif team_abbrev == away:
                return home, check_date

    return "", ""
