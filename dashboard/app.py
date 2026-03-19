# dashboard/app.py
"""
NBA Player Props Dashboard 
=============================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env so THE_ODDS_API_KEY and other secrets are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually if not installed

import pandas as pd
import numpy as np
import threading
from datetime import datetime
from dash import Dash, html, dcc, Input, Output, callback, ALL, State
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler


def get_current_nba_season():
    """
    Get the current NBA season string (e.g., '2025-26').
    NBA season starts in October, so:
    - Oct-Dec 2025 = 2025-26 season
    - Jan-Sep 2026 = still 2025-26 season
    """
    today = datetime.now()
    year = today.year
    month = today.month

    # If we're in Oct-Dec, season started this year
    # If we're in Jan-Sep, season started last year
    if month >= 10:  # October or later
        start_year = year
    else:  # January - September
        start_year = year - 1

    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def get_previous_nba_season():
    """Get the previous NBA season string."""
    current = get_current_nba_season()
    start_year = int(current.split("-")[0]) - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


# Get dynamic season values
CURRENT_SEASON = get_current_nba_season()
PREVIOUS_SEASON = get_previous_nba_season()
print(f"Current season: {CURRENT_SEASON}, Previous season: {PREVIOUS_SEASON}")

from utils.feature_engineering import engineer_features
from utils.injury_news import get_player_injury_status, get_team_injuries
from utils.prop_calculator import generate_best_props, calculate_hit_probability
from utils.data_fetch import get_todays_games, get_teams_playing_today, get_next_opponent_for_team
from utils.prop_scorer import calculate_smart_prop_score, detect_player_role
from models.predictor import NBAPredictor

# =============================================================================
# DATA LOADING (Optimized with Parquet caching)
# =============================================================================

def load_data():
    """
    Load data with Parquet caching for fast startup.

    Priority:
    1. Load pre-computed Parquet (fastest - ~2 seconds)
    2. Fall back to CSV + feature engineering (~30 seconds)
    3. Save Parquet for next startup
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    parquet_path = os.path.join(data_dir, "engineered_data.parquet")

    # Load supporting data (small files, always needed)
    team_def_path = os.path.join(data_dir, "team_defensive_stats.csv")
    team_def = pd.read_csv(team_def_path) if os.path.exists(team_def_path) else pd.DataFrame()

    positions_path = os.path.join(data_dir, "player_positions.csv")
    positions_df = pd.read_csv(positions_path) if os.path.exists(positions_path) else pd.DataFrame()

    def_vs_pos_path = os.path.join(data_dir, "defense_vs_position.csv")
    def_vs_pos = pd.read_csv(def_vs_pos_path) if os.path.exists(def_vs_pos_path) else pd.DataFrame()

    # Try to load pre-computed Parquet (FAST PATH)
    if os.path.exists(parquet_path):
        try:
            print("Loading pre-computed data (Parquet)...")
            df = pd.read_parquet(parquet_path)
            # Ensure _date column exists
            if "_date" not in df.columns:
                df["_date"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
            print(f"  Loaded {len(df)} records from Parquet (fast path)")
            return df, team_def, positions_df, def_vs_pos
        except Exception as e:
            print(f"  Parquet load failed: {e}, falling back to CSV...")

    # SLOW PATH: Load CSV and engineer features
    print("Loading from CSV (first run or cache miss)...")
    game_logs = pd.read_csv(os.path.join(data_dir, "player_game_logs.csv"))

    team_stats_path = os.path.join(data_dir, "team_stats.csv")
    team_stats = pd.read_csv(team_stats_path) if os.path.exists(team_stats_path) else pd.DataFrame()

    # Engineer features
    df = engineer_features(
        game_logs,
        team_def,
        team_stats=team_stats if not team_stats.empty else None,
        player_positions=positions_df if not positions_df.empty else None,
        defense_vs_position=def_vs_pos if not def_vs_pos.empty else None
    )
    df["_date"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df = df.drop_duplicates(subset=["PLAYER_NAME", "GAME_DATE", "MATCHUP"], keep="first")
    df = df.sort_values(["PLAYER_NAME", "_date"])

    # Save Parquet for next startup (background save)
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved Parquet cache for faster next startup")
    except Exception as e:
        print(f"  Could not save Parquet cache: {e}")

    return df, team_def, positions_df, def_vs_pos


# Lazy model loading - only load when needed
_models_cache = {}
_models_lock = threading.Lock()


def get_predictor(stat: str):
    """
    Lazy load predictor models on first use.
    This speeds up initial app startup.
    """
    stat = stat.upper()

    with _models_lock:
        if stat in _models_cache:
            return _models_cache[stat]

        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        filepath = os.path.join(models_dir, f"{stat.lower()}_predictor.pkl")

        if os.path.exists(filepath):
            print(f"Loading {stat} predictor (lazy load)...")
            _models_cache[stat] = NBAPredictor.load(filepath)
            return _models_cache[stat]

        return None


def load_models():
    """Legacy function for compatibility - now uses lazy loading."""
    # Return empty dict, models load on-demand via get_predictor()
    return {}


print("Loading data...")
DF, TEAM_DEF, PLAYER_POSITIONS, DEFENSE_VS_POS = load_data()

# Offensive team stats (team_stats.csv) — used by matchup card Team Rankings
_team_off_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "team_stats.csv")
TEAM_OFF = pd.read_csv(_team_off_path) if os.path.exists(_team_off_path) else pd.DataFrame()
if not TEAM_OFF.empty and "SEASON" in TEAM_OFF.columns:
    TEAM_OFF = TEAM_OFF[TEAM_OFF["SEASON"] == CURRENT_SEASON]

PREDICTORS = load_models()  # Empty, uses lazy loading now
PLAYERS = sorted(DF["PLAYER_NAME"].unique().tolist())

# Build player ID mapping from the data (column is "Player_ID" from NBA API)
# Build player ID mapping from the data (column is "Player_ID" from NBA API)
PLAYER_IDS = {}
if "Player_ID" in DF.columns:
    try:
        # Optimized mapping using vectorization (much faster than iterating)
        # Drop rows where Player_ID is NaN, then drop duplicates by Name
        mapping_df = DF[["PLAYER_NAME", "Player_ID"]].dropna(subset=["Player_ID"]).drop_duplicates("PLAYER_NAME")
        # Convert to dictionary
        PLAYER_IDS = dict(zip(mapping_df["PLAYER_NAME"], mapping_df["Player_ID"].astype(int)))
    except Exception as e:
        print(f"Error building player ID map: {e}")
        # Fallback to empty if fails


print(f"Loaded {len(DF)} game records for {len(PLAYERS)} players")

# Diagnostic print for Player_ID column
print(f"DEBUG: 'Player_ID' in DF.columns = {'Player_ID' in DF.columns}")

# Build player ID mapping from the data (column is "Player_ID" from NBA API)
PLAYER_IDS = {}
if "Player_ID" in DF.columns:
    for name in PLAYERS:
        player_rows = DF[DF["PLAYER_NAME"] == name]
        if len(player_rows) > 0:
            pid = player_rows["Player_ID"].iloc[0]
            if pid:
                PLAYER_IDS[name] = int(pid)

print(f"Player photos available: {len(PLAYER_IDS)}")

# ── Game predictor (spread + total) ──────────────────────────────────────────
from utils.game_predictor import GamePredictor as _GamePredictor
GAME_PREDICTOR = _GamePredictor(team_def_df=TEAM_DEF, player_logs_df=DF)
print("Game predictor initialized.")


# =============================================================================
# BACKGROUND DATA UPDATER
# =============================================================================

# Thread-safe data access
_data_lock = threading.Lock()


def get_global_df():
    """Thread-safe getter for DF."""
    with _data_lock:
        return DF.copy()


def merge_new_games(new_games_df):
    """Thread-safe merger for new games into global DF. Also updates Parquet cache."""
    global DF, PLAYER_POSITIONS
    from utils.feature_engineering import add_rolling_averages

    with _data_lock:
        try:
            # Reload PLAYER_POSITIONS (in case of trades)
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            positions_path = os.path.join(data_dir, "player_positions.csv")
            if os.path.exists(positions_path):
                PLAYER_POSITIONS = pd.read_csv(positions_path)
                print(f"[App] Roster updated: {len(PLAYER_POSITIONS)} players")

            # Add rolling averages to new games
            if not new_games_df.empty:
                new_games_df = add_rolling_averages(new_games_df)

                # Parse dates for sorting
                new_games_df["_date"] = pd.to_datetime(
                    new_games_df["GAME_DATE"],
                    format="%b %d, %Y",
                    errors="coerce"
                )

                # Merge with existing, keeping latest version of duplicates
                DF = pd.concat([DF, new_games_df]).drop_duplicates(
                    subset=["PLAYER_NAME", "GAME_DATE", "MATCHUP"],
                    keep="last"
                ).sort_values(["PLAYER_NAME", "_date"])

                print(f"[App] Global DF updated: now has {len(DF)} records")

                # Update Parquet cache for faster next startup
                try:
                    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
                    parquet_path = os.path.join(data_dir, "engineered_data.parquet")
                    df_to_save = DF.copy()
                    # Coerce numeric columns that arrive as strings from live API
                    for col in ["MIN", "PTS", "REB", "AST", "FG3M", "FGM", "FGA", "FG3A", "FTM", "FTA"]:
                        if col in df_to_save.columns:
                            df_to_save[col] = pd.to_numeric(df_to_save[col], errors="coerce")
                    df_to_save.to_parquet(parquet_path, index=False)
                    print(f"[App] Parquet cache updated")
                except Exception as pe:
                    print(f"[App] Could not update Parquet cache: {pe}")

        except Exception as e:
            print(f"[App] Error merging new games: {e}")


def scheduled_update():
    """Wrapper for scheduled data update."""
    from utils.data_updater import update_game_data
    update_game_data(get_global_df, merge_new_games)


def scheduled_props_refresh():
    """Refresh the props cache in the background."""
    from utils.props_cache import refresh_props_cache
    try:
        refresh_props_cache(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, get_predictor_fn=get_predictor)
    except Exception as e:
        print(f"[App] Props cache refresh error: {e}")


# Initialize and start background scheduler
try:
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        scheduled_update,
        'interval',
        minutes=30,
        id='game_data_updater',
        replace_existing=True,
        max_instances=1
    )
    scheduler.add_job(
        scheduled_props_refresh,
        'interval',
        minutes=30,
        id='props_cache_refresher',
        replace_existing=True,
        max_instances=1
    )
    def _scheduled_grade():
        from datetime import date as _date
        from utils.prediction_tracker import grade_predictions
        yesterday = (_date.today() - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
        today     = _date.today().strftime("%Y-%m-%d")
        for d in (yesterday, today):
            try:
                grade_predictions(d)
            except Exception as exc:
                print(f"[App] Grade predictions failed for {d}: {exc}")

    scheduler.add_job(
        _scheduled_grade,
        'cron',
        hour=1,
        minute=0,
        id='grade_predictions',
        replace_existing=True,
        max_instances=1,
    )
    scheduler.start()
    print("[App] Background schedulers started (data + props cache every 30 min, grade at 1 AM)")

    # Trigger an immediate update at startup so data is fresh on first load
    # (don't wait 30 minutes for the first scheduled run)
    import threading as _threading
    _startup_thread = _threading.Thread(target=scheduled_update, daemon=True, name="startup-update")
    _startup_thread.start()
    print("[App] Startup data refresh triggered in background")

    # Pre-warm props cache and Today's Games API caches immediately at startup.
    # This runs at module-import time so it works under gunicorn (HuggingFace)
    # as well as direct `python app.py` invocation.
    def _warmup_caches():
        # 1. Pre-fetch games + odds so first page visit hits in-memory cache
        try:
            from utils.data_fetch import get_todays_games
            from utils.odds_fetcher import get_game_odds, get_live_odds
            get_todays_games()
            get_game_odds()
            get_live_odds()
            print("[App] Today's Games + Odds caches warmed")
        except Exception as e:
            print(f"[App] Games/Odds warm failed: {e}")

        # 2. Pre-compute Best Props (most expensive — do after games are warmed)
        try:
            from utils.props_cache import refresh_props_cache
            refresh_props_cache(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, get_predictor_fn=get_predictor)
        except Exception as e:
            print(f"[App] Props cache warm failed (will retry in 30 min): {e}")

    _warmup_thread = _threading.Thread(target=_warmup_caches, daemon=True, name="startup-warmup")
    _warmup_thread.start()
    print("[App] Startup cache warmup triggered in background")
except Exception as e:
    print(f"[App] Warning: Could not start scheduler: {e}")

# =============================================================================
# COLOR SCHEME (Outlier Style)
# =============================================================================

COLORS = {
    "bg": "#060b18",
    "card": "rgba(0,0,0,0)",      # transparent — glass cards handle bg via CSS
    "border": "rgba(255,255,255,0.06)",
    "text": "#f0f4ff",
    "text_secondary": "#8ca0c0",
    "text_muted": "#4a5a75",

    # Stat colors
    "pts":  "#14b8a6",       # Teal  — points
    "ast":  "#f97066",       # Coral — assists
    "reb":  "#a78bfa",       # Violet — rebounds
    "blk":  "#60a5fa",       # Blue  — blocks
    "stl":  "#fbbf24",       # Amber — steals
    "fg3m": "#ec4899",       # Pink  — 3-pointers

    # Hit/miss
    "hit":  "#22c55e",       # Green  — above threshold
    "miss": "#f43f5e",       # Rose   — below threshold

    # Hit rate tiers
    "hit_high": "#22c55e",   # Green  66%+
    "hit_mid":  "#f59e0b",   # Amber  50-65%
    "hit_low":  "#f43f5e",   # Rose   <50%

    # Accents
    "accent":           "#14b8a6",
    "accent_secondary": "#8b5cf6",
}

# Card style — visual handled by CSS .card class; keep inline fallback minimal
CARD = {
    "borderRadius": "20px",
    "padding": "24px",
    "marginBottom": "20px",
}

def get_hit_color(pct):
    if pct >= 66:
        return COLORS["hit_high"]
    elif pct >= 50:
        return COLORS["hit_mid"]
    else:
        return COLORS["hit_low"]

def get_stat_color(stat):
    """Get the color for a stat type"""
    colors = {
        "PTS": COLORS["pts"],
        "AST": COLORS["ast"],
        "REB": COLORS["reb"],
        "BLK": COLORS["blk"],
        "STL": COLORS["stl"],
        "FG3M": COLORS["fg3m"],
    }
    return colors.get(stat, COLORS["accent"])



def get_player_current_team(player_name):
    """Get player's current team from PLAYER_POSITIONS."""
    if PLAYER_POSITIONS.empty:
        return ""
    pos_match = PLAYER_POSITIONS[PLAYER_POSITIONS["PLAYER_NAME"] == player_name]
    if len(pos_match) > 0:
        return str(pos_match["TEAM_ABBREVIATION"].iloc[0])
    return ""


def get_h2h_opponent_for_player(player_name):
    """Get opponent for a player (for H2H mode).

    Checks: today's games → most recent opponent from game log.
    Returns (opponent_abbrev, player_team)
    """
    from utils.data_fetch import extract_opponent_from_matchup

    player_team = get_player_current_team(player_name)
    if not player_team:
        return "", ""

    # Try today's games first (non-blocking — returns empty on API failure)
    try:
        teams_today = get_teams_playing_today()
        if player_team in teams_today:
            today_games = get_todays_games()
            if not today_games.empty:
                for _, game in today_games.iterrows():
                    home = game.get("HOME_TEAM", "")
                    away = game.get("AWAY_TEAM", "")
                    if player_team == home:
                        return away, player_team
                    elif player_team == away:
                        return home, player_team
    except Exception:
        pass

    # Fallback: get most recent opponent from game log (always works)
    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)
    if not player_df.empty and "MATCHUP" in player_df.columns:
        last_matchup = player_df.iloc[0].get("MATCHUP", "")
        opponent = extract_opponent_from_matchup(last_matchup)
        if opponent:
            return opponent, player_team

    return "", ""


def filter_h2h_games(player_df, h2h_mode, player_name=None):
    """Filter player games by H2H opponent if in H2H mode. Returns (filtered_df, opponent_name).

    Shows last 10 games against today's opponent.
    """
    if h2h_mode != "h2h" or len(player_df) == 0:
        return player_df, ""

    # Get player name from df if not provided
    if not player_name and len(player_df) > 0:
        player_name = player_df.iloc[0].get("PLAYER_NAME", "")

    # Get today's opponent using PLAYER_POSITIONS for current team
    opponent, _ = get_h2h_opponent_for_player(player_name)

    if not opponent:
        return player_df.head(0), ""  # Return empty df

    def is_vs_opponent(matchup):
        if not isinstance(matchup, str):
            return False
        return opponent in matchup

    # Filter to games vs this opponent, then take last 10
    filtered_df = player_df[player_df["MATCHUP"].apply(is_vs_opponent)].head(10)
    return filtered_df, opponent


# Stat type configurations (organized: main stats, combos, then other)
STAT_TYPES = [
    # Main stats first
    {"id": "PTS", "label": "PTS"},
    {"id": "AST", "label": "AST"},
    {"id": "REB", "label": "REB"},
    # Combos
    {"id": "PTS+AST", "label": "PTS+AST"},
    {"id": "PTS+REB", "label": "PTS+REB"},
    {"id": "AST+REB", "label": "AST+REB"},
    {"id": "PTS+AST+REB", "label": "PRA"},
    # Other stats
    {"id": "FG3M", "label": "3PM"},
    {"id": "BLK", "label": "BLK"},
    {"id": "STL", "label": "STL"},
    {"id": "STL+BLK", "label": "STL+BLK"},
]

# =============================================================================
# DASH APP
# =============================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


@server.route("/download-report")
def download_report():
    """Download prediction history as Excel file."""
    from flask import send_file, abort
    from utils.prediction_tracker import export_to_excel
    try:
        path = export_to_excel()
        return send_file(path, as_attachment=True, download_name="nba_predictions.xlsx")
    except Exception as e:
        print(f"[Download] Report error: {e}")
        abort(500)

# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================

# Team IDs for logos
TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764
}

def get_team_logo_url(team_abbr):
    """Get team logo URL from NBA CDN"""
    team_id = TEAM_IDS.get(team_abbr, "")
    if team_id:
        return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    return ""

def get_player_headshot_url(player_name):
    """Get NBA CDN headshot URL for a player"""
    player_id = PLAYER_IDS.get(player_name)
    if player_id:
        # standard 260x190 endpoint which is more reliable
        url = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
        # print(f"DEBUG: Player headshot URL for {player_name}: {url}")
        return url
    
    # Generic fallback
    fallback_url = "https://cdn.nba.com/headshots/nba/latest/260x190/fallback.png"
    return fallback_url

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div([
    # URL routing
    dcc.Location(id="url", refresh=False),

    # Navigation header
    html.Div([
        # Logo/Title
        html.Div("NBA Props AI", className="nav-brand"),
        
        # Navigation links
        html.Div([
            dcc.Link("Player Analysis", href="/", className="nav-link", id="nav-player"),
            dcc.Link("Today's Games", href="/games", className="nav-link", id="nav-games"),
            dcc.Link("Best Props", href="/props", className="nav-link", id="nav-props"),
        ], className="nav-links")
    ], className="nav-header"),

    # Page content container
    html.Div(id="page-content"),

])


def create_player_analysis_page():
    """Create the main player analysis page layout"""
    return html.Div([
        # Header with player photo and info
        html.Div([
            html.Div([
                # Player photo
                html.Div(id="player-photo", style={
                    "width": "84px",
                    "height": "84px",
                    "borderRadius": "50%",
                    "overflow": "hidden",
                    "marginRight": "20px",
                    "background": "linear-gradient(135deg, rgba(20,184,166,0.2), rgba(139,92,246,0.2))",
                    "border": "2px solid rgba(20,184,166,0.35)",
                    "boxShadow": "0 0 20px rgba(20,184,166,0.2)",
                    "flexShrink": "0",
                }),

                # Player info and dropdown
                html.Div([
                    dcc.Dropdown(
                        id="player-dropdown",
                        options=[{"label": p, "value": p} for p in PLAYERS],
                        value=PLAYERS[0] if PLAYERS else None,
                        placeholder="Search player...",
                        searchable=True,
                        clearable=False,
                        className="player-search-dropdown",
                        style={"width": "300px"}
                    ),
                    html.Div(id="player-header", style={"marginTop": "8px"}),
                ]),
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
        ], className="card", style={"padding": "1.5rem", "marginBottom": "2rem"}),

        # Main content
        html.Div([
            # Left panel - Main chart area
            html.Div([
                # Stat type tabs
                html.Div([
                    html.Button(
                        s["label"],
                        id=f"tab-{s['id'].lower().replace('+', '-')}",
                        n_clicks=0,
                        className="tab" + (" active" if s["id"] == "PTS" else "")
                    ) for s in STAT_TYPES
                ], className="tab-group"),

                # Store for selected stat
                dcc.Store(id="selected-stat", data="PTS"),

                # Time period tabs
                html.Div([
                    html.Button("L5", id="period-l5", n_clicks=0, className="tab"),
                    html.Button("L10", id="period-l10", n_clicks=0, className="tab active"),
                    html.Button("L20", id="period-l20", n_clicks=0, className="tab"),
                    html.Button("H2H", id="period-h2h", n_clicks=0, className="tab"),
                    html.Button("H/W", id="period-hw", n_clicks=0, className="tab"),
                    html.Button(CURRENT_SEASON.split("-")[0], id="period-current", n_clicks=0, className="tab"),
                    html.Button(PREVIOUS_SEASON.split("-")[0], id="period-previous", n_clicks=0, className="tab"),
                ], className="tab-group"),

                dcc.Store(id="selected-period", data=10),
                dcc.Store(id="selected-season", data=None),
                dcc.Store(id="selected-h2h", data=None),
                dcc.Store(id="selected-location", data=None),
                dcc.Store(id="current-season-store", data=CURRENT_SEASON),
                dcc.Store(id="previous-season-store", data=PREVIOUS_SEASON),

                # Hit rate header with avg/median
                html.Div(id="hit-rate-header", style={"marginBottom": "20px"}),

                # Threshold slider
                html.Div([
                    html.Div("Threshold:", style={
                        "color": "var(--text-secondary)",
                        "fontSize": "0.8rem",
                        "marginRight": "12px",
                        "minWidth": "70px"
                    }),
                    html.Div([
                        dcc.Slider(
                            id="threshold-slider",
                            min=0,
                            max=50,
                            step=0.5,
                            value=10,
                            marks={
                                0: {"label": "0", "style": {"color": COLORS["text_secondary"]}},
                                10: {"label": "10", "style": {"color": COLORS["text_secondary"]}},
                                20: {"label": "20", "style": {"color": COLORS["text_secondary"]}},
                                30: {"label": "30", "style": {"color": COLORS["text_secondary"]}},
                                40: {"label": "40", "style": {"color": COLORS["text_secondary"]}},
                                50: {"label": "50", "style": {"color": COLORS["text_secondary"]}},
                            },
                            included=True,
                            tooltip=None,
                        ),
                    ], style={"flex": "1", "marginRight": "16px"}),
                    html.Div(id="threshold-display", style={
                        "color": "var(--teal-400, #2dd4bf)",
                        "fontSize": "1.15rem",
                        "fontWeight": "700",
                        "minWidth": "60px",
                        "textAlign": "center",
                        "backgroundColor": "rgba(20,184,166,0.12)",
                        "border": "1px solid rgba(20,184,166,0.3)",
                        "padding": "4px 14px",
                        "borderRadius": "8px",
                        "fontFamily": "var(--font-mono, 'Fira Code', monospace)",
                    }),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

                # Main chart
                html.Div([
                    dcc.Graph(id="main-chart", config={"displayModeBar": False})
                ], className="card"),

                # Average/Median footer
                html.Div(id="avg-median-footer", style={
                    "display": "flex",
                    "justifyContent": "center",
                    "gap": "40px",
                    "padding": "12px 20px",
                    "background": "rgba(255,255,255,0.03)",
                    "border": "1px solid rgba(255,255,255,0.06)",
                    "backdropFilter": "blur(8px)",
                    "borderRadius": "12px",
                    "marginBottom": "16px"
                }),

                # Supporting Stats Section
                html.Div([
                    # Header with Average/Median toggle
                    html.Div([
                        html.Div("Supporting Stats", style={
                            "fontSize": "1.1rem",
                            "fontWeight": "700",
                        }),
                        html.Div([
                            html.Button("Average", id="supporting-avg-btn", n_clicks=1, className="btn", style={"backgroundColor": "var(--bg-tertiary)"}),
                            html.Button("Median", id="supporting-median-btn", n_clicks=0, className="btn", style={"backgroundColor": "transparent"}),
                        ])
                    ], style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "20px"
                    }),

                    # Store for average/median selection
                    dcc.Store(id="supporting-stat-mode", data="average"),

                    # Supporting stats cards
                    html.Div(id="supporting-stats-cards", style={
                        "display": "flex",
                        "gap": "12px",
                        "marginBottom": "24px",
                        "flexWrap": "wrap"
                    }),

                    # Store for selected shooting stat
                    dcc.Store(id="selected-shooting-stat", data="FG"),

                    # Shooting breakdown chart (stacked bar)
                    html.Div([
                        dcc.Graph(id="shooting-breakdown-chart", config={"displayModeBar": False})
                    ]),

                    # Insights section
                    html.Div([
                        html.Div("Insights", style={
                            "fontSize": "1.1rem",
                            "fontWeight": "700",
                            "marginBottom": "14px",
                            "marginTop": "24px",
                        }),
                        html.Div(id="player-insights", style={
                            "color": "var(--text-secondary)",
                            "fontSize": "0.95rem",
                            "lineHeight": "1.7",
                            "padding": "18px",
                            "backgroundColor": "var(--bg-primary)",
                            "borderRadius": "var(--radius-md)",
                            "borderLeft": "4px solid var(--accent-primary)"
                        })
                    ])
                ], className="card"),


            ], className="content-area"),

            # Right panel - Sidebar
            html.Div([
                # Tabs for sidebar sections
                html.Div([
                    html.Button("Matchup", id="sidebar-matchup", n_clicks=1, className="tab active"),
                    html.Button("Injuries", id="sidebar-injuries", n_clicks=0, className="tab"),
                    html.Button("Insights", id="sidebar-insights", n_clicks=0, className="tab"),
                    html.Button("Best Props", id="sidebar-props", n_clicks=0, className="tab"),
                ], className="tab-group"),

                dcc.Store(id="sidebar-tab", data="matchup"),

                # Dynamic sidebar content
                html.Div(id="sidebar-content"),

            ], className="sidebar"),

        ], className="main-container"),

        # Auto-refresh interval
        dcc.Interval(
            id="auto-refresh-interval",
            interval=5 * 60 * 1000,
            n_intervals=0
        ),

        dcc.Store(id="last-update-time", data=datetime.now().isoformat()),
    ])


def _fmt_odds(price) -> str:
    """Format American odds with sign."""
    if price is None:
        return "N/A"
    try:
        p = int(price)
        return f"+{p}" if p > 0 else str(p)
    except (TypeError, ValueError):
        return "N/A"


def _pick_badge(pick: str | None, conf: str, home_abbr: str, away_abbr: str) -> html.Div:
    """Build a model pick badge with confidence colour."""
    if not pick:
        return html.Div("—", style={"color": "var(--text-muted)", "fontSize": "0.8rem"})

    label = pick
    if pick == "HOME":
        label = f"{home_abbr} covers"
    elif pick == "AWAY":
        label = f"{away_abbr} covers"

    conf_color = {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#94a3b8"}.get(conf, "#94a3b8")
    bg         = {"HIGH": "rgba(34,197,94,0.12)", "MEDIUM": "rgba(245,158,11,0.12)", "LOW": "rgba(148,163,184,0.08)"}.get(conf, "rgba(148,163,184,0.08)")
    return html.Div([
        html.Span("⚡ ", style={"fontSize": "0.7rem"}),
        html.Span(label, style={"fontWeight": "700", "fontSize": "0.8rem"}),
        html.Div(conf, style={
            "fontSize": "0.6rem", "fontWeight": "700", "letterSpacing": "0.08em",
            "marginTop": "2px", "color": conf_color,
        }),
    ], style={
        "background": bg, "border": f"1px solid {conf_color}33",
        "borderRadius": "8px", "padding": "6px 10px",
        "color": conf_color, "textAlign": "center", "minWidth": "90px",
    })


def _form_dots(wl_list: list) -> html.Div:
    """Render last-N W/L as coloured dots."""
    dots = []
    for result in wl_list:
        color = "#22c55e" if result == "W" else "#f43f5e"
        dots.append(html.Span(style={
            "display": "inline-block", "width": "8px", "height": "8px",
            "borderRadius": "50%", "background": color,
            "boxShadow": f"0 0 4px {color}88", "marginRight": "3px", "flexShrink": "0",
        }))
    return html.Div(dots, style={"display": "flex", "alignItems": "center", "flexWrap": "nowrap"})


def _h2h_section(h2h: dict, home: str, away: str):
    """Render head-to-head record between two teams."""
    if not h2h or not h2h.get("total"):
        return None
    wins_home  = h2h.get("wins_a", 0)   # wins_a = home wins vs away
    wins_away  = h2h.get("losses_a", 0) # losses_a = away wins vs home
    total      = h2h.get("total", 0)
    if total == 0:
        return None

    if wins_home > wins_away:
        leader = f"{home} leads {wins_home}–{wins_away}"
        leader_color = "var(--teal-400, #2dd4bf)"
    elif wins_away > wins_home:
        leader = f"{away} leads {wins_away}–{wins_home}"
        leader_color = "#f59e0b"
    else:
        leader = f"Series tied {wins_home}–{wins_away}"
        leader_color = "var(--text-secondary)"

    return html.Div([
        html.Div("H2H THIS SEASON", style={
            "fontSize": "0.62rem", "fontWeight": "700", "letterSpacing": "0.1em",
            "color": "var(--text-muted)", "marginBottom": "6px",
        }),
        html.Div(leader, style={
            "fontSize": "0.88rem", "fontWeight": "700", "color": leader_color,
        }),
        html.Div(f"({total} game{'s' if total != 1 else ''})", style={
            "fontSize": "0.72rem", "color": "var(--text-muted)", "marginTop": "2px",
        }),
    ], style={
        "padding": "10px 14px",
        "background": "rgba(255,255,255,0.025)",
        "borderRadius": "10px",
        "border": "1px solid rgba(255,255,255,0.05)",
        "marginTop": "10px",
    })


def _l10_section(last10_away: list, last10_home: list, away_abbr: str, home_abbr: str):
    """Expandable last-10-games log for both teams."""
    if not last10_away and not last10_home:
        return None

    def _game_rows(games: list, abbr: str) -> list:
        if not games:
            return [html.Div("No data", style={"color": "var(--text-muted)", "fontSize": "0.75rem"})]
        rows = []
        for g in games:
            wl    = g.get("wl", "?")
            date  = g.get("date", "")[:10] if g.get("date") else "—"
            opp   = g.get("opponent", "—")
            t_pts = g.get("team_pts", "—")
            o_pts = g.get("opp_pts", "—")
            ha    = "@" if g.get("home_away") == "AWAY" else "vs"
            wl_color = "#22c55e" if wl == "W" else "#f43f5e"
            rows.append(html.Div([
                html.Span(wl, style={
                    "fontWeight": "800", "color": wl_color, "fontSize": "0.72rem",
                    "minWidth": "14px", "display": "inline-block",
                }),
                html.Span(f"{abbr} {ha} {opp}", style={
                    "fontSize": "0.72rem", "color": "var(--text-secondary)",
                    "margin": "0 8px", "minWidth": "100px", "display": "inline-block",
                }),
                html.Span(f"{t_pts}–{o_pts}", style={
                    "fontSize": "0.72rem", "fontFamily": "var(--font-mono)",
                    "color": "var(--text-muted)",
                }),
                html.Span(date, style={
                    "fontSize": "0.65rem", "color": "var(--text-muted)",
                    "marginLeft": "auto",
                }),
            ], style={
                "display": "flex", "alignItems": "center",
                "padding": "3px 0",
                "borderBottom": "1px solid rgba(255,255,255,0.03)",
            }))
        return rows

    away_rows = _game_rows(last10_away, away_abbr)
    home_rows = _game_rows(last10_home, home_abbr)

    return html.Details([
        html.Summary([
            html.Span("LAST 10 GAMES", style={
                "fontSize": "0.62rem", "fontWeight": "700", "letterSpacing": "0.1em",
                "color": "var(--text-muted)", "cursor": "pointer",
            }),
            html.Span(" ▾", style={"fontSize": "0.6rem", "color": "var(--text-muted)"}),
        ], style={"listStyle": "none", "outline": "none", "cursor": "pointer"}),

        html.Div([
            # Away team games
            html.Div([
                html.Div(away_abbr, style={
                    "fontSize": "0.65rem", "fontWeight": "700", "letterSpacing": "0.08em",
                    "color": "var(--text-secondary)", "marginBottom": "4px", "marginTop": "10px",
                }),
                *away_rows,
            ]),
            # Home team games
            html.Div([
                html.Div(home_abbr, style={
                    "fontSize": "0.65rem", "fontWeight": "700", "letterSpacing": "0.08em",
                    "color": "var(--teal-400, #2dd4bf)", "marginBottom": "4px", "marginTop": "10px",
                }),
                *home_rows,
            ]),
        ]),
    ], style={
        "padding": "10px 14px",
        "background": "rgba(255,255,255,0.02)",
        "borderRadius": "10px",
        "border": "1px solid rgba(255,255,255,0.05)",
        "marginTop": "10px",
    })


def _format_game_time(raw: str) -> str:
    """Convert ESPN ISO timestamp or status text to a readable tip-off time."""
    if not raw:
        return ""
    # If it looks like an ISO timestamp, parse it
    if "T" in raw and raw.endswith("Z"):
        try:
            from datetime import timezone
            dt_utc = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            # Convert to Eastern time (UTC-5 standard, UTC-4 daylight)
            import time as _time
            # Simple daylight saving heuristic: Mar–Nov = EDT (UTC-4), else EST (UTC-5)
            month = dt_utc.month
            offset_hours = -4 if 3 <= month <= 11 else -5
            from datetime import timedelta as _td
            dt_et = dt_utc + _td(hours=offset_hours)
            suffix = "PM" if dt_et.hour >= 12 else "AM"
            hour = dt_et.hour % 12 or 12
            tz_label = "EDT" if offset_hours == -4 else "EST"
            return f"{hour}:{dt_et.minute:02d} {suffix} {tz_label}"
        except Exception:
            pass
    # Fall back to returning the raw string (e.g., "7:30 pm ET" from ESPN status)
    return raw


def _reasoning_section(reasoning: dict, intel: list) -> "html.Div | None":
    """Render the model's pick reasoning as a collapsible section."""
    if not reasoning:
        return None

    winner_reason = reasoning.get("winner_reason", "")
    total_reason  = reasoning.get("total_reason", "")
    spread_reason = reasoning.get("spread_reason", "")
    key_factors   = reasoning.get("key_factors", [])

    rows = []
    for label, text, color in [
        ("WINNER", winner_reason, "#a78bfa"),
        ("TOTAL",  total_reason,  "#34d399"),
        ("SPREAD", spread_reason, "#60a5fa"),
    ]:
        if text:
            rows.append(html.Div([
                html.Span(label, style={
                    "fontSize": "0.58rem", "fontWeight": "700", "letterSpacing": "0.08em",
                    "color": color, "marginRight": "6px", "minWidth": "48px",
                }),
                html.Span(text, style={
                    "fontSize": "0.72rem", "color": "var(--text-secondary)", "lineHeight": "1.45",
                }),
            ], style={
                "display": "flex", "alignItems": "flex-start",
                "padding": "4px 0", "borderBottom": "1px solid rgba(255,255,255,0.04)",
            }))

    if key_factors:
        factor_chips = [
            html.Span(f"· {f}", style={
                "fontSize": "0.68rem", "color": "var(--text-muted)",
                "marginRight": "8px", "lineHeight": "1.6",
            }) for f in key_factors[:5]
        ]
        rows.append(html.Div(factor_chips, style={
            "paddingTop": "6px", "flexWrap": "wrap", "display": "flex",
        }))

    if not rows:
        return None

    return html.Details([
        html.Summary([
            html.Span("WHY THIS PICK?", style={
                "fontSize": "0.62rem", "fontWeight": "700", "letterSpacing": "0.1em",
                "color": "var(--text-muted)", "cursor": "pointer",
            }),
            html.Span(" ▾", style={"fontSize": "0.6rem", "color": "var(--text-muted)"}),
        ], style={"listStyle": "none", "outline": "none", "cursor": "pointer"}),
        html.Div(rows, style={"marginTop": "8px"}),
    ], style={
        "padding": "10px 14px",
        "background": "rgba(167,139,250,0.03)",
        "borderRadius": "10px",
        "border": "1px solid rgba(167,139,250,0.10)",
        "marginTop": "10px",
    })


def _injury_panel(home_team: str, away_team: str):
    """Render injury report for both teams in a collapsible dropdown."""
    _STATUS_COLOR = {
        "OUT":          ("#f43f5e", "rgba(244,63,94,0.12)",  "rgba(244,63,94,0.25)"),
        "DOUBTFUL":     ("#f97316", "rgba(249,115,22,0.10)", "rgba(249,115,22,0.25)"),
        "QUESTIONABLE": ("#f59e0b", "rgba(245,158,11,0.08)", "rgba(245,158,11,0.2)"),
        "PROBABLE":     ("#a3e635", "rgba(163,230,53,0.06)", "rgba(163,230,53,0.18)"),
    }

    def _player_chip(player: dict):
        status = player["status"]
        color, bg, border = _STATUS_COLOR.get(status, ("#94a3b8", "rgba(148,163,184,0.06)", "rgba(148,163,184,0.15)"))
        reason = player.get("reason", "")
        return html.Div([
            html.Span(player["name"], style={
                "fontSize": "0.78rem", "fontWeight": "600",
                "color": "var(--text-primary)", "marginRight": "6px",
            }),
            html.Span(status, style={
                "fontSize": "0.6rem", "fontWeight": "700", "letterSpacing": "0.06em",
                "color": color, "background": bg, "border": f"1px solid {border}",
                "borderRadius": "4px", "padding": "1px 5px",
            }),
            html.Span(f" · {reason}", style={
                "fontSize": "0.7rem", "color": "var(--text-muted)",
                "marginLeft": "4px", "fontStyle": "italic",
            }) if reason else None,
        ], style={
            "display": "flex", "alignItems": "center", "flexWrap": "wrap",
            "padding": "4px 0", "borderBottom": "1px solid rgba(255,255,255,0.03)",
        })

    def _team_block(abbr: str):
        try:
            players = get_team_injuries(abbr)
        except Exception:
            players = []

        if not players:
            return html.Div([
                html.Span(abbr, style={"fontSize": "0.65rem", "fontWeight": "700", "color": "var(--text-secondary)", "marginRight": "8px"}),
                html.Span("No injuries reported", style={"fontSize": "0.72rem", "color": "var(--text-muted)", "fontStyle": "italic"}),
            ], style={"marginBottom": "6px"})

        return html.Div([
            html.Div(abbr, style={
                "fontSize": "0.65rem", "fontWeight": "700", "letterSpacing": "0.08em",
                "color": "var(--text-secondary)", "marginBottom": "4px",
            }),
            *[_player_chip(p) for p in players],
        ], style={"marginBottom": "8px"})

    # Count total injured players for the badge
    try:
        total_injured = len(get_team_injuries(away_team)) + len(get_team_injuries(home_team))
    except Exception:
        total_injured = 0
    badge = f" ({total_injured})" if total_injured else ""

    return html.Details([
        html.Summary([
            html.Span(f"INJURY REPORT{badge}", style={
                "fontSize": "0.62rem", "fontWeight": "700", "letterSpacing": "0.1em",
                "color": "var(--text-muted)", "cursor": "pointer",
            }),
            html.Span(" ▾", style={"fontSize": "0.6rem", "color": "var(--text-muted)"}),
        ], style={"listStyle": "none", "outline": "none", "cursor": "pointer"}),
        html.Div([
            _team_block(away_team),
            _team_block(home_team),
        ], style={"marginTop": "8px"}),
    ], style={
        "padding": "10px 14px",
        "background": "rgba(244,63,94,0.03)",
        "borderRadius": "10px",
        "border": "1px solid rgba(244,63,94,0.08)",
        "marginTop": "10px",
    })


def create_todays_games_page():
    """Premium Today's Games page: team matchup cards with live odds + model predictions."""
    from utils.data_fetch import get_todays_games
    from utils.odds_fetcher import get_game_odds

    games     = get_todays_games()
    game_odds = get_game_odds()   # dict keyed "AWAY@HOME"

    _PAGE_TITLE_STYLE = {
        "fontSize": "1.6rem", "fontWeight": "800", "letterSpacing": "-0.02em",
        "background": "linear-gradient(135deg, #f0f4ff 40%, #8ca0c0 100%)",
        "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
        "backgroundClip": "text", "marginBottom": "4px",
    }

    if games.empty:
        return html.Div([
            html.Div([
                html.Div("TODAY'S GAMES", style=_PAGE_TITLE_STYLE),
                html.Div(datetime.now().strftime("%A, %B %d, %Y"),
                         style={"color": "var(--text-muted)", "marginBottom": "32px", "fontSize": "0.9rem"}),
                html.Div([
                    html.Div("No games scheduled", style={"fontSize": "1.1rem", "fontWeight": "600", "marginBottom": "8px"}),
                    html.Div("Schedule data temporarily unavailable. Check nba.com/schedule.",
                             style={"color": "var(--text-muted)", "fontSize": "0.9rem"}),
                    html.Div("Player Analysis & Best Props are fully functional.",
                             style={"color": "var(--accent-primary)", "fontSize": "0.9rem", "marginTop": "8px"}),
                ], style={"textAlign": "center", "padding": "60px 0"})
            ], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "2rem"})
        ])

    # ── Build game cards ─────────────────────────────────────────────────────
    # ESPN/API abbreviation → internal data abbreviation (must match game_logs/positions CSVs)
    _ABBR_FIX = {"SAS": "SAN"}

    # Refresh predictor ONCE before the loop (not per-game — avoids N×DF copy)
    GAME_PREDICTOR.refresh(get_global_df())

    game_cards = []
    _predictions_for_tracker: list[dict] = []  # accumulate for save_daily_predictions
    for _, game in games.iterrows():
        home_team   = game.get("HOME_TEAM", game.get("HOME_TEAM_ABBREVIATION", ""))
        away_team   = game.get("AWAY_TEAM", game.get("VISITOR_TEAM_ABBREVIATION", game.get("VISITOR_TEAM", "")))
        game_status = game.get("GAME_STATUS_TEXT", "Scheduled")

        # Normalize abbreviations for internal data lookups (predictor, form, etc.)
        home_team_internal = _ABBR_FIX.get(home_team, home_team)
        away_team_internal = _ABBR_FIX.get(away_team, away_team)

        home_logo = get_team_logo_url(home_team)
        away_logo = get_team_logo_url(away_team)

        # ── Odds ─────────────────────────────────────────────────────────────
        odds_key  = f"{away_team}@{home_team}"
        odds      = game_odds.get(odds_key, {})
        spread    = odds.get("spread") or {}
        total     = odds.get("total")  or {}
        h2h       = odds.get("h2h")    or {}
        bookmaker = odds.get("bookmaker", "")

        spread_home_line  = spread.get("home_line")
        spread_away_line  = spread.get("away_line")
        spread_home_price = spread.get("home_price")
        spread_away_price = spread.get("away_price")
        total_line        = total.get("line")
        over_price        = total.get("over_price")
        under_price       = total.get("under_price")
        home_ml           = h2h.get("home_price")
        away_ml           = h2h.get("away_price")
        has_odds          = bool(spread or total)

        # Format spread labels
        def _fmt_spread(line):
            if line is None: return "N/A"
            return f"+{line}" if line > 0 else str(line)

        # ── Model prediction (injury-aware) ──────────────────────────────────
        try:
            home_inj = get_team_injuries(home_team_internal)
            away_inj = get_team_injuries(away_team_internal)
            prediction = GAME_PREDICTOR.predict_game(
                home_team_internal, away_team_internal,
                home_injuries=home_inj,
                away_injuries=away_inj,
            )
            pick       = GAME_PREDICTOR.get_pick(
                prediction,
                home=home_team_internal,
                away=away_team_internal,
                actual_spread=spread_home_line,
                actual_total=total_line,
            )
            last10_home = GAME_PREDICTOR.get_team_last_n_games(home_team_internal, 10)
            last10_away = GAME_PREDICTOR.get_team_last_n_games(away_team_internal, 10)
            h2h_data    = prediction.get("h2h", {})
        except Exception as e:
            logger.warning(f"GamePredictor error for {home_team} vs {away_team}: {e}")
            prediction = {"predicted_home_score": None, "predicted_away_score": None,
                          "predicted_total": None, "predicted_spread": None,
                          "winner": None, "winner_margin": None, "winner_confidence": "LOW",
                          "home_form": {}, "away_form": {}, "h2h": {}, "intel": [],
                          "reasoning": {}}
            pick = {"spread_pick": None, "spread_confidence": "LOW",
                    "total_pick": None, "total_confidence": "LOW",
                    "winner_pick": None, "winner_confidence": "LOW",
                    "model_spread": None, "model_total": None}
            last10_home, last10_away, h2h_data = [], [], {}

        home_form  = prediction.get("home_form", {})
        away_form  = prediction.get("away_form", {})
        intel      = prediction.get("intel", [])
        reasoning  = prediction.get("reasoning", {})
        pred_home  = prediction.get("predicted_home_score")
        pred_away  = prediction.get("predicted_away_score")
        pred_total = prediction.get("predicted_total")

        # Reverse-map internal abbrevs back to display abbrevs for pick badges
        _ABBR_REVERSE = {v: k for k, v in _ABBR_FIX.items()}
        if pick.get("winner_pick") and pick["winner_pick"] in _ABBR_REVERSE:
            pick["winner_pick"] = _ABBR_REVERSE[pick["winner_pick"]]
        if pick.get("spread_team") and pick["spread_team"] in _ABBR_REVERSE:
            pick["spread_team"] = _ABBR_REVERSE[pick["spread_team"]]

        # Collect for daily prediction tracking
        _predictions_for_tracker.append({
            "game_id":   f"{away_team}@{home_team}",
            "home":      home_team,
            "away":      away_team,
            "game_time": game.get("GAME_TIME", game.get("GAME_STATUS_TEXT", "")),
            "predictions": {
                "winner_pick":        pick.get("winner_pick"),
                "winner_confidence":  pick.get("winner_confidence"),
                "spread_pick":        pick.get("spread_pick"),
                "spread_team":        pick.get("spread_team"),
                "spread_line":        spread_home_line,
                "spread_confidence":  pick.get("spread_confidence"),
                "total_pick":         pick.get("total_pick"),
                "total_line":         total_line,
                "total_confidence":   pick.get("total_confidence"),
                "model_home_score":   pred_home,
                "model_away_score":   pred_away,
                "model_total":        pred_total,
                "reasoning":          reasoning,
            },
        })

        # ── Status badge ─────────────────────────────────────────────────────
        is_live    = "Q" in str(game_status) or "Halftime" in str(game_status)
        status_el = html.Span([
            html.Span(style={
                "display": "inline-block", "width": "6px", "height": "6px",
                "borderRadius": "50%", "background": "#22c55e",
                "marginRight": "5px", "animation": "pulse-dot 1.8s ease-in-out infinite",
            }) if is_live else None,
            html.Span(game_status),
        ], className="game-time-chip",
           style={"borderColor": "rgba(34,197,94,0.3)", "color": "#22c55e"} if is_live else {})

        # ── Odds columns ─────────────────────────────────────────────────────
        def _odds_col(label, main_content, sub_content, model_el):
            return html.Div([
                html.Div(label, className="odds-col-label"),
                html.Div(main_content, className="odds-col-main"),
                html.Div(sub_content, className="odds-col-sub"),
                html.Div(style={"height": "1px", "background": "rgba(255,255,255,0.05)", "margin": "8px 0"}),
                html.Div("MODEL PICK", className="odds-col-model-label"),
                model_el,
            ], className="odds-col")

        # Spread column
        if spread_home_line is not None:
            spread_main = html.Div([
                html.Div([
                    html.Span(away_team, style={"color": "var(--text-muted)", "fontSize": "0.7rem", "marginRight": "4px"}),
                    html.Span(_fmt_spread(spread_away_line), style={"fontWeight": "700", "color": "var(--text-primary)"}),
                ], style={"marginBottom": "2px"}),
                html.Div([
                    html.Span(home_team, style={"color": "var(--text-muted)", "fontSize": "0.7rem", "marginRight": "4px"}),
                    html.Span(_fmt_spread(spread_home_line), style={"fontWeight": "700", "color": "var(--text-primary)"}),
                ]),
            ])
            spread_sub = html.Div([
                html.Span(f"{_fmt_odds(spread_away_price)} / {_fmt_odds(spread_home_price)}",
                          style={"fontSize": "0.72rem", "color": "var(--text-muted)", "fontFamily": "var(--font-mono)"}),
            ])
        else:
            spread_main = html.Div("—", style={"color": "var(--text-muted)"})
            spread_sub  = html.Div()

        # Total column
        if total_line is not None:
            total_main = html.Div([
                html.Span("O/U ", style={"color": "var(--text-muted)", "fontSize": "0.75rem"}),
                html.Span(f"{total_line}", style={"fontWeight": "700", "fontSize": "1.1rem",
                                                   "fontFamily": "var(--font-mono)", "color": "var(--text-primary)"}),
            ])
            total_sub = html.Div([
                html.Span(f"O {_fmt_odds(over_price)}  U {_fmt_odds(under_price)}",
                          style={"fontSize": "0.72rem", "color": "var(--text-muted)", "fontFamily": "var(--font-mono)"}),
            ])
        else:
            total_main = html.Div("—", style={"color": "var(--text-muted)"})
            total_sub  = html.Div()

        # Moneyline column
        if home_ml is not None:
            ml_main = html.Div([
                html.Div([
                    html.Span(away_team, style={"color": "var(--text-muted)", "fontSize": "0.7rem", "marginRight": "4px"}),
                    html.Span(_fmt_odds(away_ml), style={"fontWeight": "700",
                        "color": "#22c55e" if (away_ml or 0) > 0 else "var(--text-primary)"}),
                ], style={"marginBottom": "2px"}),
                html.Div([
                    html.Span(home_team, style={"color": "var(--text-muted)", "fontSize": "0.7rem", "marginRight": "4px"}),
                    html.Span(_fmt_odds(home_ml), style={"fontWeight": "700",
                        "color": "#22c55e" if (home_ml or 0) > 0 else "var(--text-primary)"}),
                ]),
            ])
            ml_sub = html.Div(
                bookmaker, style={"fontSize": "0.65rem", "color": "var(--text-muted)", "textTransform": "uppercase", "letterSpacing": "0.06em"})
        else:
            ml_main = html.Div("—", style={"color": "var(--text-muted)"})
            ml_sub  = html.Div()

        # Model total pick label
        total_pick = pick.get("total_pick")
        if total_pick and total_line:
            total_pick_label = f"{total_pick} {total_line}"
        elif pred_total:
            total_pick_label = f"Model: {pred_total:.0f}"
        else:
            total_pick_label = None

        spread_pick_label = pick.get("spread_pick")
        if spread_pick_label == "HOME" and spread_home_line is not None:
            spread_pick_label = f"{home_team} {_fmt_spread(spread_home_line)}"
        elif spread_pick_label == "AWAY" and spread_away_line is not None:
            spread_pick_label = f"{away_team} {_fmt_spread(spread_away_line)}"

        # ── Form rows ────────────────────────────────────────────────────────
        home_wl   = home_form.get("wl_list", [])
        away_wl   = away_form.get("wl_list", [])
        home_wins = home_form.get("wins", 0)
        home_loss = home_form.get("losses", 0)
        away_wins = away_form.get("wins", 0)
        away_loss = away_form.get("losses", 0)
        home_rpg  = home_form.get("rolling_ppg")
        away_rpg  = away_form.get("rolling_ppg")

        def _form_row(abbr, wl_list, wins, losses, rpg):
            return html.Div([
                html.Span(abbr, style={
                    "fontSize": "0.7rem", "fontWeight": "700", "minWidth": "30px",
                    "color": "var(--text-secondary)", "marginRight": "8px",
                }),
                _form_dots(wl_list),
                html.Span(f"({wins}-{losses})", style={
                    "fontSize": "0.72rem", "color": "var(--text-muted)",
                    "marginLeft": "8px", "fontFamily": "var(--font-mono)",
                }),
                html.Span(f"  {rpg:.0f} PPG", style={
                    "fontSize": "0.72rem", "color": "var(--teal-400, #2dd4bf)",
                    "marginLeft": "8px", "fontFamily": "var(--font-mono)",
                }) if rpg else None,
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"})

        # ── Assemble card ─────────────────────────────────────────────────────
        game_card = html.Div([

            # ── ROW 1: Status + logo + league label ──────────────────────────
            html.Div([
                html.Div([
                    html.Span("NBA", style={
                        "fontSize": "0.65rem", "fontWeight": "700", "letterSpacing": "0.1em",
                        "color": "var(--text-muted)", "background": "rgba(255,255,255,0.05)",
                        "border": "1px solid rgba(255,255,255,0.08)", "borderRadius": "4px",
                        "padding": "2px 7px", "marginRight": "10px",
                    }),
                    status_el,
                ], style={"display": "flex", "alignItems": "center"}),
                html.Div(
                    _format_game_time(game.get("GAME_TIME", game.get("GAME_STATUS_TEXT", ""))) if not is_live else "",
                    style={"fontSize": "0.75rem", "color": "var(--text-muted)"},
                ),
            ], style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "marginBottom": "16px",
            }),

            # ── ROW 2: Teams + logos ─────────────────────────────────────────
            html.Div([
                # Away team
                html.Div([
                    html.Img(src=away_logo, style={
                        "width": "60px", "height": "60px", "objectFit": "contain",
                        "filter": "drop-shadow(0 2px 8px rgba(0,0,0,0.5))",
                    }) if away_logo else html.Div(away_team[0], style={
                        "width": "60px", "height": "60px", "borderRadius": "50%",
                        "background": "var(--bg-tertiary)", "display": "flex",
                        "alignItems": "center", "justifyContent": "center",
                        "fontSize": "1.4rem", "fontWeight": "800",
                    }),
                    html.Div(away_team, style={
                        "fontWeight": "800", "fontSize": "1.3rem", "marginTop": "6px",
                        "letterSpacing": "-0.02em", "color": "var(--text-primary)",
                    }),
                    html.Div("AWAY", style={
                        "fontSize": "0.62rem", "fontWeight": "600", "letterSpacing": "0.08em",
                        "color": "var(--text-muted)", "marginTop": "1px",
                    }),
                ], style={"textAlign": "center", "flex": "1"}),

                # VS divider
                html.Div([
                    html.Div("VS", style={
                        "fontSize": "0.7rem", "fontWeight": "700", "letterSpacing": "0.12em",
                        "color": "var(--text-muted)", "background": "rgba(255,255,255,0.04)",
                        "border": "1px solid rgba(255,255,255,0.07)", "borderRadius": "50%",
                        "width": "36px", "height": "36px", "display": "flex",
                        "alignItems": "center", "justifyContent": "center",
                    }),
                ], style={"display": "flex", "alignItems": "center", "padding": "0 16px"}),

                # Home team
                html.Div([
                    html.Img(src=home_logo, style={
                        "width": "60px", "height": "60px", "objectFit": "contain",
                        "filter": "drop-shadow(0 2px 8px rgba(0,0,0,0.5))",
                    }) if home_logo else html.Div(home_team[0], style={
                        "width": "60px", "height": "60px", "borderRadius": "50%",
                        "background": "var(--bg-tertiary)", "display": "flex",
                        "alignItems": "center", "justifyContent": "center",
                        "fontSize": "1.4rem", "fontWeight": "800",
                    }),
                    html.Div(home_team, style={
                        "fontWeight": "800", "fontSize": "1.3rem", "marginTop": "6px",
                        "letterSpacing": "-0.02em", "color": "var(--text-primary)",
                    }),
                    html.Div("HOME", style={
                        "fontSize": "0.62rem", "fontWeight": "600", "letterSpacing": "0.08em",
                        "color": "var(--teal-400, #2dd4bf)", "marginTop": "1px",
                    }),
                ], style={"textAlign": "center", "flex": "1"}),
            ], style={
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "padding": "8px 0 20px",
                "borderBottom": "1px solid rgba(255,255,255,0.05)",
                "marginBottom": "16px",
            }),

            # ── ROW 3: Odds grid (Spread | Total | Moneyline) ────────────────
            html.Div([
                _odds_col(
                    "SPREAD",
                    spread_main, spread_sub,
                    _pick_badge(spread_pick_label, pick.get("spread_confidence", "LOW"), home_team, away_team)
                    if pick.get("spread_pick") else html.Div([
                        html.Div(f"Model: {pred_home:.0f}-{pred_away:.0f}" if pred_home else "—",
                                 style={"fontSize": "0.78rem", "color": "var(--text-muted)", "textAlign": "center"}),
                    ]),
                ),
                html.Div(style={"width": "1px", "background": "rgba(255,255,255,0.05)", "margin": "0 4px", "alignSelf": "stretch"}),
                _odds_col(
                    "TOTAL",
                    total_main, total_sub,
                    _pick_badge(total_pick_label, pick.get("total_confidence", "LOW"), home_team, away_team)
                    if pick.get("total_pick") else html.Div(
                        f"Model: {pred_total:.0f}" if pred_total else "—",
                        style={"fontSize": "0.78rem", "color": "var(--text-muted)", "textAlign": "center"},
                    ),
                ),
                html.Div(style={"width": "1px", "background": "rgba(255,255,255,0.05)", "margin": "0 4px", "alignSelf": "stretch"}),
                _odds_col(
                    "MONEYLINE",
                    ml_main, ml_sub,
                    _pick_badge(pick.get("winner_pick"), pick.get("winner_confidence", "LOW"), home_team, away_team)
                    if pick.get("winner_pick") else html.Div([
                        html.Div(f"Model: {pred_home:.0f}-{pred_away:.0f}" if pred_home else "—",
                                 style={"fontSize": "0.78rem", "color": "var(--text-muted)", "textAlign": "center"}),
                    ]),
                ),
            ], className="odds-grid"),

            # ── ROW 4: Team Form ──────────────────────────────────────────────
            html.Div([
                html.Div("RECENT FORM", style={
                    "fontSize": "0.62rem", "fontWeight": "700", "letterSpacing": "0.1em",
                    "color": "var(--text-muted)", "marginBottom": "8px",
                }),
                _form_row(away_team, away_wl, away_wins, away_loss, away_rpg),
                _form_row(home_team, home_wl, home_wins, home_loss, home_rpg),
            ], style={
                "padding": "12px 14px",
                "background": "rgba(255,255,255,0.025)",
                "borderRadius": "10px",
                "border": "1px solid rgba(255,255,255,0.05)",
                "marginTop": "14px",
            }) if (home_wl or away_wl) else None,

            # ── ROW 5: Model Reasoning ────────────────────────────────────────
            _reasoning_section(reasoning, intel),

            # ── ROW 6: H2H Record ─────────────────────────────────────────────
            _h2h_section(h2h_data, home_team, away_team),

            # ── ROW 7: L10 Game Log (expandable) ─────────────────────────────
            _l10_section(last10_away, last10_home, away_team, home_team),

            # ── ROW 8: Injury Report (collapsible dropdown) ───────────────────
            _injury_panel(home_team, away_team),

        ], className="card game-card-v2")
        game_cards.append(game_card)

    # Save today's predictions asynchronously (fire-and-forget)
    if _predictions_for_tracker:
        import threading as _t
        _date_str = datetime.now().strftime("%Y-%m-%d")
        _t.Thread(
            target=lambda: __import__("utils.prediction_tracker", fromlist=["save_daily_predictions"])
                           .save_daily_predictions(_date_str, _predictions_for_tracker),
            daemon=True, name="save-predictions",
        ).start()

    return html.Div([
        html.Div([
            html.Div([
                html.Div("TODAY'S GAMES", style=_PAGE_TITLE_STYLE),
                html.A(
                    "⬇ Download Report",
                    href="/download-report",
                    style={
                        "fontSize": "0.72rem", "fontWeight": "600", "letterSpacing": "0.05em",
                        "color": "var(--teal-400, #2dd4bf)", "border": "1px solid var(--teal-400, #2dd4bf)",
                        "borderRadius": "6px", "padding": "5px 12px", "textDecoration": "none",
                        "opacity": "0.8",
                    },
                ),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "4px"}),
            html.Div([
                html.Span(datetime.now().strftime("%A, %B %d, %Y"),
                          style={"color": "var(--text-muted, #4a5a75)", "fontSize": "0.9rem"}),
                html.Span(f" • {len(games)} games",
                          style={"color": "var(--teal-400, #2dd4bf)", "marginLeft": "8px", "fontSize": "0.9rem", "fontWeight": "600"}),
                html.Span(" • Odds + Model Predictions",
                          style={"color": "var(--text-muted, #4a5a75)", "marginLeft": "4px", "fontSize": "0.85rem"}),
            ], style={"marginBottom": "24px"}),
            html.Div(game_cards, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(440px, 1fr))", "gap": "20px"}),
        ], style={"maxWidth": "1300px", "margin": "0 auto", "padding": "2rem"})
    ])


def create_best_props_page():
    """Create the Best Props page showing top value picks (reads from pre-computed cache)."""
    from utils.props_cache import get_cached_props

    cache = get_cached_props()
    props_data = cache["main_page_data"]
    has_todays_games = cache["has_todays_games"]
    game_matchups = cache["game_matchups"]
    target_date = cache.get("target_date")

    # Format the target date label (today vs tomorrow)
    from datetime import date as _date
    today_str = _date.today().isoformat()
    if target_date and target_date != today_str:
        from datetime import datetime as _dt
        try:
            label_date = _dt.strptime(target_date, "%Y-%m-%d").strftime("%A, %B %d, %Y")
            date_note = f"Tomorrow's Slate — {label_date}"
        except ValueError:
            date_note = target_date
    else:
        label_date = datetime.now().strftime("%A, %B %d, %Y")
        date_note = label_date

    return html.Div([
        html.Div([
            # Page header
            html.Div([
                html.Div("BEST PROPS" if has_todays_games else "BEST PROPS", style={
                    "fontSize": "1.6rem", "fontWeight": "800", "marginBottom": "6px",
                    "letterSpacing": "-0.02em",
                    "background": "linear-gradient(135deg, #f0f4ff 40%, #8ca0c0 100%)",
                    "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                    "backgroundClip": "text"
                }),
                html.Div([
                    html.Span(date_note, style={"color": "var(--text-muted, #4a5a75)", "fontSize": "0.9rem"}),
                    html.Span(f" • {len(props_data)} picks", style={"color": "var(--teal-400, #2dd4bf)", "marginLeft": "8px", "fontSize": "0.9rem", "fontWeight": "600"})
                ], style={"marginBottom": "6px"}),
                html.Div("Top value player props ranked by Expected Value — upcoming games only", style={"color": "var(--text-secondary, #8ca0c0)", "fontSize": "0.875rem", "marginBottom": "24px"}),
            ]),

            # Row 1: Stat type filter
            html.Div([
                html.Div("All Stats",  id="props-filter-all",  n_clicks=0, className="tab active"),
                html.Div("Points",     id="props-filter-pts",  n_clicks=0, className="tab"),
                html.Div("Assists",    id="props-filter-ast",  n_clicks=0, className="tab"),
                html.Div("Rebounds",   id="props-filter-reb",  n_clicks=0, className="tab"),
                html.Div("3-Pointers", id="props-filter-3pm",  n_clicks=0, className="tab"),
                html.Div("Pts+Ast",    id="props-filter-pa",   n_clicks=0, className="tab"),
                html.Div("Pts+Reb",    id="props-filter-pr",   n_clicks=0, className="tab"),
                html.Div("Ast+Reb",    id="props-filter-ar",   n_clicks=0, className="tab"),
                html.Div("PTS+AST+REB",id="props-filter-pra",  n_clicks=0, className="tab"),
                html.Div("🔒 LOCKS",   id="props-filter-lock", n_clicks=0, className="tab"),
            ], className="tab-group", style={"marginBottom": "12px", "flexWrap": "wrap"}),

            # Row 2: Location | Game | Sort — all on one line
            html.Div([
                html.Div("All", id="props-loc-all",  n_clicks=0, className="tab active"),
                html.Div("Home",id="props-loc-home", n_clicks=0, className="tab"),
                html.Div("Away",id="props-loc-away", n_clicks=0, className="tab"),

                html.Div(style={"flex": "1"}),  # spacer

                html.Div([
                    html.Span("Game:", style={"color": "var(--text-muted)", "fontSize": "0.85rem", "marginRight": "8px", "whiteSpace": "nowrap"}),
                    dcc.Dropdown(
                        id="props-game-filter",
                        options=[{"label": "All Games", "value": "all"}] + [{"label": m, "value": m} for m in game_matchups],
                        value="all",
                        clearable=False,
                        style={"width": "185px"},
                        className="game-filter-dropdown"
                    ),
                ], style={"display": "flex" if game_matchups else "none", "alignItems": "center", "gap": "0"}),

                html.Div([
                    html.Span("Sort:", style={"color": "var(--text-muted)", "fontSize": "0.85rem", "marginRight": "8px", "whiteSpace": "nowrap"}),
                    dcc.Dropdown(
                        id="props-sort-dropdown",
                        options=[
                            {"label": "Highest EV",       "value": "ev"},
                            {"label": "Highest Hit Rate", "value": "hit_rate"}
                        ],
                        value="ev",
                        clearable=False,
                        style={"width": "160px"},
                        className="game-filter-dropdown"
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "0"}),

            ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "24px", "flexWrap": "wrap"}),

            dcc.Store(id="props-data-store", data=props_data),
            dcc.Store(id="props-location-filter", data="all"),
            dcc.Store(id="props-stat-filter", data="all"),

            # Prop cards container (populated by callback)
            html.Div(id="props-list"),

            html.Div(f"Last updated: {datetime.now().strftime('%I:%M %p')}", style={"color": "var(--text-muted)", "fontSize": "0.8rem", "textAlign": "center", "marginTop": "24px"})

        ], style={"maxWidth": "800px", "margin": "0 auto"})
    ])


# =============================================================================
# PAGE ROUTING
# =============================================================================

@callback(
    [Output("page-content", "children"),
     Output("nav-player", "className"),
     Output("nav-games", "className"),
     Output("nav-props", "className")],
    Input("url", "pathname")
)
def display_page(pathname):
    """Route to the correct page based on URL"""
    active = "nav-link active"
    inactive = "nav-link"

    if pathname == "/games":
        return create_todays_games_page(), inactive, active, inactive
    elif pathname == "/props":
        return create_best_props_page(), inactive, inactive, active
    else:
        return create_player_analysis_page(), active, inactive, inactive


# =============================================================================
# CALLBACKS
# =============================================================================

# Tab click handlers for stat type
@callback(
    Output("selected-stat", "data"),
    [Input(f"tab-{s['id'].lower().replace('+', '-')}", "n_clicks") for s in STAT_TYPES]
)
def update_stat_selection(*clicks):
    from dash import ctx
    triggered = ctx.triggered_id

    if triggered is None:
        return "PTS"

    for s in STAT_TYPES:
        tab_id = f"tab-{s['id'].lower().replace('+', '-')}"
        if triggered == tab_id:
            return s["id"]

    return "PTS"


# Update tab styles based on selection
@callback(
    [Output(f"tab-{s['id'].lower().replace('+', '-')}", "className") for s in STAT_TYPES],
    Input("selected-stat", "data")
)
def update_stat_tab_styles(selected):
    styles = []
    for s in STAT_TYPES:
        base_class = "tab"
        if s["id"] == selected:
            styles.append(f"{base_class} active")
        else:
            styles.append(base_class)
    return styles


# Tab click handlers for period
@callback(
    [Output("selected-period", "data"),
     Output("selected-season", "data"),
     Output("selected-h2h", "data"),
     Output("selected-location", "data"),
     Output("period-l5", "className"),
     Output("period-l10", "className"),
     Output("period-l20", "className"),
     Output("period-h2h", "className"),
     Output("period-hw", "className"),
     Output("period-hw", "children"),
     Output("period-current", "className"),
     Output("period-previous", "className")],
    [Input("period-l5", "n_clicks"),
     Input("period-l10", "n_clicks"),
     Input("period-l20", "n_clicks"),
     Input("period-h2h", "n_clicks"),
     Input("period-hw", "n_clicks"),
     Input("period-current", "n_clicks"),
     Input("period-previous", "n_clicks")],
    [State("selected-location", "data")]
)
def update_period_tabs(l5, l10, l20, h2h, hw, current, previous, current_location):
    from dash import ctx
    triggered = ctx.triggered_id

    # Default all to inactive
    styles = ["tab"] * 7
    period = 10
    season = None
    h2h_mode = None
    location = None
    hw_label = "H/W"

    if triggered == "period-l5":
        period, styles[0] = 5, "tab active"
    elif triggered == "period-l10" or triggered is None:
        period, styles[1] = 10, "tab active"
    elif triggered == "period-l20":
        period, styles[2] = 20, "tab active"
    elif triggered == "period-h2h":
        period, h2h_mode, styles[3] = 100, "h2h", "tab active"
    elif triggered == "period-hw":
        if current_location is None:
            location, hw_label = "home", "Home"
        elif current_location == "home":
            location, hw_label = "away", "Away"
        else:
            location, hw_label = None, "H/W"
        
        period = 100 if location else 10
        styles[4] = "tab active" if location else "tab"
        
        # If toggling off, go back to L10
        if not location:
            styles[1] = "tab active"
            
    elif triggered == "period-current":
        period, season, styles[5] = 100, CURRENT_SEASON, "tab active"
    elif triggered == "period-previous":
        period, season, styles[6] = 100, PREVIOUS_SEASON, "tab active"

    return [period, season, h2h_mode, location] + styles[:4] + [styles[4], hw_label] + styles[5:]


# Sidebar tab handler
@callback(
    [Output("sidebar-tab", "data"),
     Output("sidebar-matchup", "className"),
     Output("sidebar-injuries", "className"),
     Output("sidebar-insights", "className"),
     Output("sidebar-props", "className")],
    [Input("sidebar-matchup", "n_clicks"),
     Input("sidebar-injuries", "n_clicks"),
     Input("sidebar-insights", "n_clicks"),
     Input("sidebar-props", "n_clicks")]
)
def update_sidebar_tabs(matchup, injuries, insights, props):
    from dash import ctx
    triggered = ctx.triggered_id

    styles = ["tab"] * 4
    tab = "matchup"

    if triggered == "sidebar-matchup" or triggered is None:
        tab, styles[0] = "matchup", "tab active"
    elif triggered == "sidebar-injuries":
        tab, styles[1] = "injuries", "tab active"
    elif triggered == "sidebar-insights":
        tab, styles[2] = "insights", "tab active"
    elif triggered == "sidebar-props":
        tab, styles[3] = "props", "tab active"

    return [tab] + styles



# =============================================================================
# BEST PROPS PAGE CALLBACKS
# =============================================================================

@callback(
    [Output("props-location-filter", "data"),
     Output("props-loc-all", "className"),
     Output("props-loc-home", "className"),
     Output("props-loc-away", "className")],
    [Input("props-loc-all", "n_clicks"),
     Input("props-loc-home", "n_clicks"),
     Input("props-loc-away", "n_clicks")],
    prevent_initial_call=True
)
def update_location_filter(all_clicks, home_clicks, away_clicks):
    from dash import ctx
    triggered = ctx.triggered_id

    active = "tab active"
    inactive = "tab"

    if triggered == "props-loc-home":
        return "home", inactive, active, inactive
    elif triggered == "props-loc-away":
        return "away", inactive, inactive, active
    else:  # all or default
        return "all", active, inactive, inactive


@callback(
    [Output("props-stat-filter", "data"),
     Output("props-filter-all",  "className"),
     Output("props-filter-pts",  "className"),
     Output("props-filter-ast",  "className"),
     Output("props-filter-reb",  "className"),
     Output("props-filter-3pm",  "className"),
     Output("props-filter-pa",   "className"),
     Output("props-filter-pr",   "className"),
     Output("props-filter-ar",   "className"),
     Output("props-filter-pra",  "className"),
     Output("props-filter-lock", "className")],
    [Input("props-filter-all",  "n_clicks"),
     Input("props-filter-pts",  "n_clicks"),
     Input("props-filter-ast",  "n_clicks"),
     Input("props-filter-reb",  "n_clicks"),
     Input("props-filter-3pm",  "n_clicks"),
     Input("props-filter-pa",   "n_clicks"),
     Input("props-filter-pr",   "n_clicks"),
     Input("props-filter-ar",   "n_clicks"),
     Input("props-filter-pra",  "n_clicks"),
     Input("props-filter-lock", "n_clicks")],
    prevent_initial_call=True,
)
def update_stat_filter(*_):
    from dash import ctx
    triggered = ctx.triggered_id
    active   = "tab active"
    inactive = "tab"
    n = 10  # total number of tabs
    def _cls(active_idx):
        return [active if i == active_idx else inactive for i in range(n)]
    mapping = {
        "props-filter-pts":  ("PTS",         1),
        "props-filter-ast":  ("AST",         2),
        "props-filter-reb":  ("REB",         3),
        "props-filter-3pm":  ("FG3M",        4),
        "props-filter-pa":   ("PTS+AST",     5),
        "props-filter-pr":   ("PTS+REB",     6),
        "props-filter-ar":   ("AST+REB",     7),
        "props-filter-pra":  ("PTS+AST+REB", 8),
        "props-filter-lock": ("LOCK",        9),
    }
    if triggered in mapping:
        val, idx = mapping[triggered]
        return [val] + _cls(idx)
    return ["all"] + _cls(0)


@callback(
    [Output("player-dropdown", "value"),
     Output("url", "pathname")],
    Input({"type": "prop-player-name", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_player_from_prop(n_clicks_list):
    """Navigate to player analysis page when a prop card player name is clicked."""
    from dash import ctx
    if not any(n_clicks_list):
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        return triggered["index"], "/"
    from dash.exceptions import PreventUpdate
    raise PreventUpdate


def _build_odds_row(prop: dict, stat_color: str, avg: float, location_label: str, hit_pct: int) -> list:
    """
    Build the bottom stat/line row for a prop card.

    When live sportsbook odds are available (has_live_odds=True), shows:
      • The real bookmaker line and over/under prices
      • A badge with the bookmaker name (e.g. "FanDuel")

    Falls back to the estimated line when no live odds are loaded.
    """
    line       = prop.get("line", 0)
    has_live   = prop.get("has_live_odds", False)
    bookmaker  = prop.get("live_bookmaker", "")
    over_price = prop.get("live_over_price")
    under_price= prop.get("live_under_price")

    def _fmt_american(p):
        if p is None:
            return ""
        return f"+{p}" if p > 0 else str(p)

    if has_live:
        odds_badge = html.Div([
            html.Span(bookmaker, style={
                "fontSize": "0.65rem", "fontWeight": "700", "color": "var(--accent-primary)",
                "border": "1px solid var(--accent-primary)", "borderRadius": "4px",
                "padding": "1px 5px", "marginRight": "6px"
            }),
            html.Span(
                f"O {_fmt_american(over_price)}  /  U {_fmt_american(under_price)}",
                style={"fontSize": "0.75rem", "color": "var(--text-muted)"}
            ),
        ], style={"display": "flex", "alignItems": "center"})
    else:
        odds_badge = html.Span(
            "est. line",
            style={"fontSize": "0.65rem", "color": "var(--text-muted)",
                   "border": "1px solid var(--bg-tertiary)", "borderRadius": "4px", "padding": "1px 5px"}
        )

    row = html.Div([
        html.Div([
            html.Span(prop.get("stat", ""), style={
                "color": stat_color, "fontWeight": "700", "fontSize": "0.9rem", "marginRight": "8px"
            }),
            html.Span(f"Over {line}", style={"fontWeight": "700", "fontSize": "1.1rem", "marginRight": "8px"}),
            odds_badge,
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(f"L10 avg: {avg}  •  {location_label} {hit_pct}%",
                 style={"color": "var(--text-muted)", "fontSize": "0.85rem"})
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "backgroundColor": "var(--bg-primary)", "padding": "10px 12px", "borderRadius": "var(--radius-sm)"
    })
    return [row]


@callback(
    Output("props-list", "children"),
    [Input("props-location-filter", "data"),
     Input("props-game-filter", "value"),
     Input("props-sort-dropdown", "value"),
     Input("props-data-store", "data"),
     Input("props-stat-filter", "data")]
)
def update_props_list(location_filter, game_filter, sort_by, props_data, stat_filter):
    if not props_data:
        return html.Div("No props data available", style={
            "color": "var(--text-muted)",
            "textAlign": "center",
            "padding": "40px"
        })

    # Filter by stat type
    if stat_filter and stat_filter != "all":
        if stat_filter == "LOCK":
            props_data = [p for p in props_data if p.get("is_lock")]
        else:
            props_data = [p for p in props_data if p.get("stat") == stat_filter]

    # Filter props based on game
    game_filtered = []
    for prop in props_data:
        if game_filter == "all" or not game_filter:
            game_filtered.append(prop)
        elif prop.get("game_matchup", "") == game_filter:
            game_filtered.append(prop)

    # Then filter by location
    filtered_props = []
    for prop in game_filtered:
        if location_filter == "all":
            filtered_props.append(prop)
        elif location_filter == "home" and prop.get("is_home_today", True):
            filtered_props.append(prop)
        elif location_filter == "away" and not prop.get("is_home_today", True):
            filtered_props.append(prop)

    # Sort Logic
    # 1. Determine the metric key based on filter context
    hit_rate_key = "hit_rate"
    if location_filter == "home":
        hit_rate_key = "hit_rate_home"
    elif location_filter == "away":
        hit_rate_key = "hit_rate_away"

    # 2. Apply sort
    if sort_by == "hit_rate":
        filtered_props.sort(key=lambda x: x.get(hit_rate_key, 0), reverse=True)
    else: # Default to EV
        filtered_props.sort(key=lambda x: x.get("ev", 0), reverse=True)

    if not filtered_props:
        return html.Div(f"No {'home' if location_filter == 'home' else 'away' if location_filter == 'away' else ''} props found", style={
            "color": "var(--text-muted)",
            "textAlign": "center",
            "padding": "40px"
        })

    # Build prop cards — no arbitrary cap, show everything that passed filters
    prop_cards = []
    for prop in filtered_props:
        # Choose hit rate based on filter
        if location_filter == "home":
            hit_rate = prop.get("hit_rate_home", 0)
            hits = prop.get("hits_home", 0)
            total = prop.get("total_home", 0)
            avg = prop.get("avg_home", prop.get("avg", 0))
        elif location_filter == "away":
            hit_rate = prop.get("hit_rate_away", 0)
            hits = prop.get("hits_away", 0)
            total = prop.get("total_away", 0)
            avg = prop.get("avg_away", prop.get("avg", 0))
        else:
            hit_rate = prop.get("hit_rate", 0)
            hits = prop.get("hits", 0)
            total = prop.get("total", 0)
            avg = prop.get("avg", 0)

        hit_pct = int(hit_rate * 100)
        # Dynamic color for hit rate
        hit_color = "var(--success)" if hit_pct >= 70 else "var(--warning)" if hit_pct >= 60 else "var(--text-primary)"

        stat_colors = {
            "PTS": "var(--accent-primary)", # Teal
            "AST": "#f97066", # Coral
            "REB": "#a78bfa", # Purple
            "FG3M": "#ec4899" # Pink
        }
        stat_color = stat_colors.get(prop.get("stat", "PTS"), "var(--text-primary)")

        is_home_today = prop.get("is_home_today", True)
        player_id = PLAYER_IDS.get(prop.get("player", ""), "")
        player_photo = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png" if player_id else ""
        
        # Format EV
        ev_val = prop.get("ev", 0)
        ev_text = f"+{ev_val:.1f}% EV" if ev_val > 0 else f"{ev_val:.1f}% EV"
        ev_color = "var(--success)" if ev_val > 0 else "var(--text-muted)"

        # Build hit-rate insight line
        location_label = "At Home" if location_filter == "home" else "On Road" if location_filter == "away" else "Overall"

        # Pull insight data from pre-computed cache
        _raw_insight = prop.get("insight") or {}
        if isinstance(_raw_insight, str):
            insight_data = {"narrative": _raw_insight, "factors": [], "matchup_grade": "C", "trend": "neutral"}
        else:
            insight_data = _raw_insight
        narrative = insight_data.get("narrative", "")
        factors = insight_data.get("factors", [])
        matchup_grade = insight_data.get("matchup_grade", "C")
        trend = insight_data.get("trend", "neutral")

        # Matchup grade badge colors
        grade_colors = {"A": "var(--success)", "B": "#a3e635", "C": "var(--text-muted)", "D": "var(--warning)"}
        grade_color = grade_colors.get(matchup_grade, "var(--text-muted)")

        # Factors (up to 3 most relevant)
        factor_items = []
        for f in factors[:3]:
            positive = f.get("positive")
            dot_color = "var(--success)" if positive is True else "var(--warning)" if positive is False else "var(--text-muted)"
            factor_items.append(
                html.Div([
                    html.Span("● ", style={"color": dot_color, "fontSize": "0.6rem", "verticalAlign": "middle", "marginRight": "6px"}),
                    html.Span(f.get("text", ""), style={"color": "var(--text-secondary)", "fontSize": "0.85rem"})
                ], style={"marginBottom": "3px"})
            )

        prop_card = html.Div([
            # ── Header row: photo / name / matchup badge / hit% / EV ──────────
            html.Div([
                html.Div([
                    html.Img(src=player_photo, style={
                        "width": "52px", "height": "52px", "borderRadius": "50%",
                        "objectFit": "cover", "backgroundColor": "var(--bg-tertiary)",
                        "border": "2px solid rgba(20,184,166,0.3)",
                        "boxShadow": "0 0 10px rgba(20,184,166,0.15)"
                    }) if player_photo else html.Div(style={
                        "width": "52px", "height": "52px", "borderRadius": "50%",
                        "background": "linear-gradient(135deg, rgba(20,184,166,0.15), rgba(139,92,246,0.15))",
                        "border": "2px solid rgba(20,184,166,0.25)"
                    })
                ], style={"marginRight": "14px", "flexShrink": "0"}),
                html.Div([
                    html.Div([
                        html.Span(
                            prop.get("player", ""),
                            id={"type": "prop-player-name", "index": prop.get("player", "")},
                            n_clicks=0,
                            style={"fontWeight": "700", "fontSize": "1.1rem", "cursor": "pointer",
                                   "borderBottom": "1px dashed var(--text-muted)"}
                        ),
                        html.Span(f" {'vs' if is_home_today else '@'} {prop.get('opponent', '')}", style={"color": "var(--text-muted)", "fontSize": "0.9rem", "marginLeft": "8px"}),
                        html.Span(f" {'HOME' if is_home_today else 'AWAY'}", style={
                            "color": "var(--accent-primary)" if is_home_today else "var(--text-secondary)",
                            "fontSize": "0.7rem", "fontWeight": "600", "marginLeft": "8px",
                            "padding": "2px 6px", "backgroundColor": "var(--bg-primary)", "borderRadius": "4px"
                        }),
                        html.Span(f" {matchup_grade}", style={
                            "color": grade_color, "fontSize": "0.7rem", "fontWeight": "700",
                            "marginLeft": "6px", "padding": "2px 6px",
                            "border": f"1px solid {grade_color}", "borderRadius": "4px"
                        }) if matchup_grade != "C" else None,
                        html.Span("LOCK", style={
                            "fontSize": "0.65rem", "fontWeight": "800", "letterSpacing": "0.06em",
                            "color": "#fff", "background": "linear-gradient(90deg,#f59e0b,#ef4444)",
                            "borderRadius": "4px", "padding": "2px 7px", "marginLeft": "8px",
                        }) if prop.get("is_lock") else None,
                        html.Span("COMBO", style={
                            "fontSize": "0.6rem", "fontWeight": "700", "letterSpacing": "0.06em",
                            "color": "#a78bfa", "border": "1px solid #a78bfa",
                            "borderRadius": "4px", "padding": "1px 5px", "marginLeft": "6px",
                        }) if prop.get("is_combo") else None,
                    ]),
                    html.Div(f"{prop.get('team', '')} • {prop.get('position', '')}", style={"fontSize": "0.85rem", "color": "var(--text-muted)"})
                ], style={"flex": "1"}),

                # Hit Rate + EV column
                html.Div([
                    html.Div(f"{hit_pct}%", style={"fontSize": "1.5rem", "fontWeight": "800", "color": hit_color, "textAlign": "right"}),
                    html.Div(ev_text, style={"fontSize": "0.8rem", "fontWeight": "600", "color": ev_color, "textAlign": "right"})
                ])
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),

            # ── Narrative insight ──────────────────────────────────────────────
            html.Div(narrative, style={
                "color": "var(--text-secondary)", "fontSize": "0.88rem", "lineHeight": "1.5",
                "marginBottom": "10px", "fontStyle": "italic"
            }) if narrative else None,

            # ── Factors list ──────────────────────────────────────────────────
            html.Div(factor_items, style={
                "background": "rgba(34,197,94,0.06)", "padding": "10px 12px",
                "borderRadius": "8px", "borderLeft": "3px solid var(--success)",
                "border": "1px solid rgba(34,197,94,0.15)", "marginBottom": "10px"
            }) if factor_items else html.Div([
                html.Div([
                    html.Span("● ", style={"color": "var(--success)", "fontSize": "0.6rem", "verticalAlign": "middle", "marginRight": "6px"}),
                    html.Span(f"{location_label}: {hits}/{total} ({hit_pct}%)", style={"color": "var(--text-secondary)", "fontSize": "0.85rem"})
                ])
            ], style={
                "background": "rgba(34,197,94,0.06)", "padding": "10px 12px",
                "borderRadius": "8px", "borderLeft": "3px solid var(--success)",
                "border": "1px solid rgba(34,197,94,0.15)", "marginBottom": "10px"
            }),

            # ── Stat / Line / Avg row ─────────────────────────────────────────
            *_build_odds_row(prop, stat_color, avg, location_label, hit_pct),
        ], className="card prop-card")
        prop_cards.append(prop_card)

    return prop_cards


@callback(
    Output("player-photo", "children"),
    Input("player-dropdown", "value")
)
def update_player_photo(player_name):
    if not player_name:
        return None

    headshot_url = get_player_headshot_url(player_name)

    return html.Img(
        src=headshot_url,
        style={
            "width": "100%",
            "height": "100%",
            "objectFit": "cover",
        }
    )


@callback(
    Output("player-header", "children"),
    Input("player-dropdown", "value")
)
def update_player_header(player_name):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name]
    current = player_df[player_df["SEASON"] == CURRENT_SEASON]
    if current.empty:
        current = player_df

    # Get team from matchup
    last_game = player_df.iloc[-1] if len(player_df) > 0 else None
    team = ""
    if last_game is not None and "MATCHUP" in player_df.columns:
        matchup = last_game.get("MATCHUP", "")
        if isinstance(matchup, str):
            team = matchup.split()[0] if matchup else ""

    avg_pts = current["PTS"].mean()
    avg_ast = current["AST"].mean()
    avg_reb = current["REB"].mean()

    return html.Div([
        html.Div([
            html.Span(player_name, style={
                "fontSize": "20px",
                "fontWeight": "600",
                "marginRight": "12px"
            }),
            html.Span(team, style={
                "fontSize": "14px",
                "color": COLORS["text_secondary"],
                "backgroundColor": COLORS["border"],
                "padding": "2px 8px",
                "borderRadius": "4px"
            }) if team else None,
        ]),
        html.Div([
            html.Span(f"{avg_pts:.1f} PTS", style={"color": COLORS["pts"], "marginRight": "16px", "fontWeight": "500", "fontSize": "13px"}),
            html.Span(f"{avg_ast:.1f} AST", style={"color": COLORS["ast"], "marginRight": "16px", "fontWeight": "500", "fontSize": "13px"}),
            html.Span(f"{avg_reb:.1f} REB", style={"color": COLORS["reb"], "fontWeight": "500", "fontSize": "13px"}),
        ], style={"marginTop": "4px"})
    ])


@callback(
    Output("threshold-slider", "value"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data")]
)
def update_threshold_default(player_name, stat):
    """Set default threshold based on player's average"""
    if not player_name:
        return 10

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    if "+" in stat:
        parts = stat.split("+")
        vals = sum(player_df[p] for p in parts if p in player_df.columns)
    else:
        vals = player_df[stat] if stat in player_df.columns else pd.Series([0])

    avg = vals.head(20).mean()
    return round(avg * 2) / 2  # Round to nearest 0.5


@callback(
    Output("threshold-display", "children"),
    Input("threshold-slider", "value")
)
def update_threshold_display(value):
    """Display the current threshold value"""
    return f"{value}"


@callback(
    Output("hit-rate-header", "children"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data"),
     Input("selected-period", "data"),
     Input("selected-h2h", "data"),
     Input("selected-location", "data"),
     Input("threshold-slider", "value")]
)
def update_hit_rate_header(player_name, stat, period, h2h_mode, location, threshold):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # H2H mode: filter by today's opponent (last 10 games vs today's opponent)
    h2h_opponent = ""
    if h2h_mode == "h2h":
        player_df, h2h_opponent = filter_h2h_games(player_df, h2h_mode, player_name)
    # Location filter: filter by home or away games
    elif location == "home":
        if "MATCHUP" in player_df.columns:
            player_df = player_df[player_df["MATCHUP"].str.contains("vs.", na=False)]
    elif location == "away":
        if "MATCHUP" in player_df.columns:
            player_df = player_df[player_df["MATCHUP"].str.contains("@", na=False)]

    # Calculate stat values
    if "+" in stat:
        parts = stat.split("+")
        player_df["_stat"] = sum(player_df[p] for p in parts if p in player_df.columns)
    else:
        player_df["_stat"] = player_df[stat] if stat in player_df.columns else 0

    # Calculate hit rates for different periods
    def calc_hit(df, n):
        recent = df.head(n)
        if len(recent) == 0:
            return 0, 0
        hits = (recent["_stat"] >= threshold).sum()
        return (hits / len(recent)) * 100, hits

    # Get display name for stat
    stat_display = stat.replace("+", " + ")
    for s in STAT_TYPES:
        if s["id"] == stat:
            stat_display = s["label"]
            break

    # H2H mode display
    if h2h_mode == "h2h":
        total_games = len(player_df)
        if total_games == 0:
            return html.Div([
                html.Div([
                    html.Span("% ", style={"color": COLORS["accent"], "fontSize": "18px"}),
                    html.Span(f"{player_name} - {stat_display}", style={"fontSize": "16px", "fontWeight": "500"}),
                ], style={"marginBottom": "12px"}),
                html.Div(f"No games found vs {h2h_opponent}" if h2h_opponent else "Player not playing today",
                         style={"color": COLORS["text_secondary"]})
            ])

        hit_pct, hits = calc_hit(player_df, total_games)

        return html.Div([
            html.Div([
                html.Span("% ", style={"color": COLORS["accent"], "fontSize": "18px"}),
                html.Span(f"{player_name} - {stat_display}", style={"fontSize": "16px", "fontWeight": "500"}),
                html.Span(f" vs {h2h_opponent}", style={"color": COLORS["accent"], "fontSize": "14px", "marginLeft": "8px"}),
            ], style={"marginBottom": "12px"}),

            html.Div([
                # Main hit rate
                html.Div([
                    html.Div(f"All {total_games} games vs {h2h_opponent}", style={"color": COLORS["text_secondary"], "fontSize": "12px"}),
                    html.Div([
                        html.Span(f"{hit_pct:.0f}%", style={
                            "color": get_hit_color(hit_pct),
                            "fontSize": "24px",
                            "fontWeight": "600"
                        }),
                        html.Span(f" {hits} of {total_games}", style={
                            "color": COLORS["text_secondary"],
                            "fontSize": "14px",
                            "marginLeft": "8px"
                        })
                    ])
                ], style={"marginRight": "40px"}),
            ], style={"display": "flex", "alignItems": "flex-end"}),
        ])

    # Normal mode display
    total_games = len(player_df)
    display_games = min(10, total_games) if location else min(period, total_games)
    hit_pct, hits = calc_hit(player_df, display_games)
    l5_pct, _ = calc_hit(player_df, 5)
    l20_pct, _ = calc_hit(player_df, 20)

    # Season hit rates (dynamic)
    df_current = player_df[player_df["SEASON"] == CURRENT_SEASON]
    df_previous = player_df[player_df["SEASON"] == PREVIOUS_SEASON]
    pct_current, _ = calc_hit(df_current, len(df_current)) if len(df_current) > 0 else (0, 0)
    pct_previous, _ = calc_hit(df_previous, len(df_previous)) if len(df_previous) > 0 else (0, 0)

    # Determine label for main display
    if location == "home":
        main_label = f"Home ({display_games} games)"
    elif location == "away":
        main_label = f"Away ({display_games} games)"
    else:
        main_label = f"Last {display_games}"

    return html.Div([
        html.Div([
            html.Span("% ", style={"color": COLORS["accent"], "fontSize": "18px"}),
            html.Span(f"{player_name} - {stat_display}", style={"fontSize": "16px", "fontWeight": "500"}),
        ], style={"marginBottom": "12px"}),

        html.Div([
            # Main hit rate
            html.Div([
                html.Div(main_label, style={"color": COLORS["text_secondary"], "fontSize": "12px"}),
                html.Div([
                    html.Span(f"{hit_pct:.0f}%", style={
                        "color": get_hit_color(hit_pct),
                        "fontSize": "24px",
                        "fontWeight": "600"
                    }),
                    html.Span(f" {hits} of {display_games}", style={
                        "color": COLORS["text_secondary"],
                        "fontSize": "14px",
                        "marginLeft": "8px"
                    })
                ])
            ], style={"marginRight": "40px"}),

            # Other periods
            html.Div([
                html.Div("L5", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{l5_pct:.0f}%", style={"color": get_hit_color(l5_pct), "fontWeight": "600"})
            ], style={"textAlign": "center", "marginRight": "16px"}),

            html.Div([
                html.Div("L20", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{l20_pct:.0f}%", style={"color": get_hit_color(l20_pct), "fontWeight": "600"})
            ], style={"textAlign": "center", "marginRight": "16px"}),

            html.Div([
                html.Div(CURRENT_SEASON.split("-")[0], style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{pct_current:.0f}%", style={"color": get_hit_color(pct_current), "fontWeight": "600"})
            ], style={"textAlign": "center", "marginRight": "16px"}),

            html.Div([
                html.Div(PREVIOUS_SEASON.split("-")[0], style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{pct_previous:.0f}%", style={"color": get_hit_color(pct_previous), "fontWeight": "600"})
            ], style={"textAlign": "center"}),

        ], style={"display": "flex", "alignItems": "flex-end"}),
    ])


@callback(
    Output("avg-median-footer", "children"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data"),
     Input("selected-period", "data"),
     Input("selected-h2h", "data"),
     Input("selected-location", "data")]
)
def update_avg_median(player_name, stat, period, h2h_mode, location):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # H2H mode: filter by today's opponent (last 10 games vs today's opponent)
    h2h_opponent = ""
    if h2h_mode == "h2h":
        player_df, h2h_opponent = filter_h2h_games(player_df, h2h_mode, player_name)
    # Location filter: filter by home or away games
    elif location == "home":
        if "MATCHUP" in player_df.columns:
            player_df = player_df[player_df["MATCHUP"].str.contains("vs.", na=False)].head(10)
        else:
            player_df = player_df.head(10)
    elif location == "away":
        if "MATCHUP" in player_df.columns:
            player_df = player_df[player_df["MATCHUP"].str.contains("@", na=False)].head(10)
        else:
            player_df = player_df.head(10)
    else:
        player_df = player_df.head(period)

    if "+" in stat:
        parts = stat.split("+")
        vals = sum(player_df[p] for p in parts if p in player_df.columns)
    else:
        vals = player_df[stat] if stat in player_df.columns else pd.Series([0])

    avg = vals.mean() if len(vals) > 0 else 0
    median = vals.median() if len(vals) > 0 else 0

    # Determine label
    if h2h_mode == "h2h" and h2h_opponent:
        label = f"vs {h2h_opponent}"
    elif location == "home":
        label = "Home"
    elif location == "away":
        label = "Away"
    else:
        label = f"L{period}"

    return [
        html.Div([
            html.Span(f"Average ({label})", style={"color": COLORS["text_secondary"], "marginRight": "8px"}),
            html.Span(f"{avg:.1f}", style={"fontWeight": "600"})
        ]),
        html.Div([
            html.Span(f"Median ({label})", style={"color": COLORS["text_secondary"], "marginRight": "8px"}),
            html.Span(f"{median:.1f}", style={"fontWeight": "600"})
        ])
    ]


@callback(
    Output("main-chart", "figure"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data"),
     Input("selected-period", "data"),
     Input("selected-season", "data"),
     Input("selected-h2h", "data"),
     Input("selected-location", "data"),
     Input("threshold-slider", "value")]
)
def update_main_chart(player_name, stat, period, season, h2h_mode, location, threshold):
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # H2H mode: filter by today's opponent (last 10 games vs today's opponent)
    if h2h_mode == "h2h":
        player_df, _ = filter_h2h_games(player_df, h2h_mode, player_name)
    # Location filter: filter by home or away games
    elif location == "home":
        # Home games have "vs." in matchup
        if "MATCHUP" in player_df.columns:
            player_df = player_df[player_df["MATCHUP"].str.contains("vs.", na=False)].head(10)
        else:
            player_df = player_df.head(10)
    elif location == "away":
        # Away games have "@" in matchup
        if "MATCHUP" in player_df.columns:
            player_df = player_df[player_df["MATCHUP"].str.contains("@", na=False)].head(10)
        else:
            player_df = player_df.head(10)
    # Filter by season if specified
    elif season:
        player_df = player_df[player_df["SEASON"] == season]
    else:
        player_df = player_df.head(period)

    player_df = player_df.iloc[::-1]  # Reverse for chronological order

    fig = go.Figure()

    # Determine if stacked or single
    is_stacked = "+" in stat

    if is_stacked:
        parts = stat.split("+")

        # Calculate totals to determine hit/miss colors
        totals = []
        for _, row in player_df.iterrows():
            total = sum(row.get(p, 0) for p in parts)
            totals.append(total)

        # Add stacked bars with hit/miss coloring
        for i, part in enumerate(parts):
            part_color = get_stat_color(part)

            # Create colors based on whether total hits threshold
            bar_colors = []
            for total in totals:
                if total >= threshold:
                    bar_colors.append(part_color)  # Use stat color for hits
                else:
                    bar_colors.append(COLORS["miss"])  # Red for misses

            # For stacked bars, only color the top bar red/green to indicate hit/miss
            if i == len(parts) - 1:  # Last (top) segment
                fig.add_trace(go.Bar(
                    x=list(range(len(player_df))),
                    y=player_df[part],
                    name=part,
                    marker_color=bar_colors,
                    text=[f"{int(row[part])}<br><span style='font-size:9px'>{part}</span>" for _, row in player_df.iterrows()],
                    textposition="inside",
                    textfont=dict(size=11, color="white"),
                    hovertemplate=f"{part}: %{{y}}<extra></extra>"
                ))
            else:
                fig.add_trace(go.Bar(
                    x=list(range(len(player_df))),
                    y=player_df[part],
                    name=part,
                    marker_color=part_color,
                    text=[f"{int(row[part])}<br><span style='font-size:9px'>{part}</span>" for _, row in player_df.iterrows()],
                    textposition="inside",
                    textfont=dict(size=11, color="white"),
                    hovertemplate=f"{part}: %{{y}}<extra></extra>"
                ))

        # Add total labels on top
        fig.add_trace(go.Scatter(
            x=list(range(len(player_df))),
            y=[t + 1.5 for t in totals],
            mode="text",
            text=[f"{int(v)}" for v in totals],
            textposition="top center",
            textfont=dict(size=12, color=COLORS["text"]),
            showlegend=False,
            hoverinfo="skip"
        ))

    else:
        # Single stat bars with hit/miss coloring
        values = player_df[stat].tolist() if stat in player_df.columns else [0] * len(player_df)

        stat_color = get_stat_color(stat)
        bar_colors = [stat_color if v >= threshold else COLORS["miss"] for v in values]

        fig.add_trace(go.Bar(
            x=list(range(len(player_df))),
            y=values,
            marker_color=bar_colors,
            text=[f"{int(v)}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate=f"{stat}: %{{y}}<extra></extra>"
        ))

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color=COLORS["accent"],
        line_width=1.5,
        annotation_text=f"{threshold}",
        annotation_position="left",
        annotation_font_color=COLORS["accent"]
    )

    # X-axis labels (date + opponent)
    labels = []
    for _, row in player_df.iterrows():
        date_str = row["_date"].strftime("%-m/%d") if pd.notna(row["_date"]) else ""
        matchup = row.get("MATCHUP", "")
        if isinstance(matchup, str):
            if "@" in matchup:
                opp = "@ " + matchup.split("@")[-1].strip()[:3]
            elif "vs." in matchup:
                opp = "vs " + matchup.split("vs.")[-1].strip()[:3]
            else:
                opp = ""
        else:
            opp = ""
        labels.append(f"{date_str}<br>{opp}")

    fig.update_layout(
        barmode="stack" if is_stacked else "relative",
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=50, r=20, t=20, b=60),
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(player_df))),
            ticktext=labels,
            tickfont=dict(size=10, color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        height=400,
        bargap=0.3,
    )

    return fig


@callback(
    Output("sidebar-content", "children"),
    [Input("sidebar-tab", "data"),
     Input("player-dropdown", "value"),
     Input("selected-stat", "data")]
)
def update_sidebar_content(tab, player_name, stat):
    if tab == "props":
        return create_best_props_content(stat)

    if not player_name:
        return None

    if tab == "matchup":
        return create_matchup_content(player_name, stat)
    elif tab == "injuries":
        return create_injuries_content(player_name)
    else:
        return create_insights_content(player_name, stat)


# =============================================================================
# SUPPORTING STATS CALLBACKS
# =============================================================================

@callback(
    [Output("supporting-stat-mode", "data"),
     Output("supporting-avg-btn", "className"),
     Output("supporting-median-btn", "className")],
    [Input("supporting-avg-btn", "n_clicks"),
     Input("supporting-median-btn", "n_clicks")]
)
def update_supporting_stat_mode(_avg_clicks, _median_clicks):
    from dash import ctx
    triggered = ctx.triggered_id

    active = "btn active"
    inactive = "btn"

    if triggered == "supporting-median-btn":
        return "median", inactive, active
    else:
        return "average", active, inactive


@callback(
    [Output("supporting-stats-cards", "children"),
     Output("selected-shooting-stat", "data")],
    [Input("player-dropdown", "value"),
     Input("supporting-stat-mode", "data"),
     Input("selected-period", "data"),
     Input("selected-season", "data"),
     Input("selected-shooting-stat", "data")]
)
def update_supporting_stats_cards(player_name, mode, period, season, current_selected):
    """Supporting stats should NOT change with H2H - always show period/season stats."""
    from dash import ctx

    if not player_name:
        return [], "FG"

    # Determine selected stat - reset to FG on player change, otherwise keep current
    triggered_id = ctx.triggered_id if ctx.triggered_id else "player-dropdown"
    if triggered_id == "player-dropdown":
        selected_stat = "FG"
    else:
        selected_stat = current_selected if current_selected else "FG"

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # Filter by season or period (H2H does NOT affect supporting stats)
    if season:
        player_df = player_df[player_df["SEASON"] == season]
    elif period:
        player_df = player_df.head(period)

    if len(player_df) == 0:
        return [], "FG"

    # Calculate stats
    def calc_stat(col):
        if col not in player_df.columns:
            return 0
        if mode == "median":
            return player_df[col].median()
        return player_df[col].mean()

    # Stats to display - ALL are now selectable/clickable
    stats = [
        {"label": "Potential Ast", "value": calc_stat("AST"), "key": "AST", "col": "AST", "type": "simple"},
        {"label": "Minutes", "value": calc_stat("MIN"), "key": "MIN", "col": "MIN", "type": "simple"},
        {"label": "Fouls", "value": calc_stat("PF") if "PF" in player_df.columns else 0, "key": "PF", "col": "PF", "type": "simple"},
        {
            "label": "Field Goals",
            "value": calc_stat("FGM"),
            "pct": (calc_stat("FGM") / calc_stat("FGA") * 100) if calc_stat("FGA") > 0 else 0,
            "attempts": calc_stat("FGA"),
            "key": "FG",
            "type": "shooting"
        },
        {
            "label": "3pts",
            "value": calc_stat("FG3M"),
            "pct": (calc_stat("FG3M") / calc_stat("FG3A") * 100) if calc_stat("FG3A") > 0 else 0,
            "attempts": calc_stat("FG3A"),
            "key": "FG3",
            "type": "shooting"
        },
        {
            "label": "Free Throws",
            "value": calc_stat("FTM") if "FTM" in player_df.columns else 0,
            "pct": (calc_stat("FTM") / calc_stat("FTA") * 100) if "FTA" in player_df.columns and calc_stat("FTA") > 0 else 0,
            "attempts": calc_stat("FTA") if "FTA" in player_df.columns else 0,
            "key": "FT",
            "type": "shooting"
        },
    ]

    # Build stat cards
    cards = []

    for stat in stats:
        is_selected = stat["key"] == selected_stat
        card_class = "stat-card active" if is_selected else "stat-card"

        if stat["type"] == "shooting":
            # Shooting stat with percentage
            card_content = html.Div([
                html.Div(stat["label"], style={"fontSize": "0.75rem", "fontWeight": "600", "color": "var(--text-secondary)", "marginBottom": "4px"}),
                html.Div(f"{stat['value']:.1f} Made ({stat['pct']:.0f}%)", style={"fontSize": "0.85rem", "fontWeight": "500"}),
                html.Div(f"{stat['attempts']:.1f} Attempts", style={"fontSize": "0.7rem", "color": "var(--text-muted)"}),
            ], id={"type": "supporting-stat-btn", "index": stat["key"]}, n_clicks=0, className=card_class)
        else:
            # Simple stat (Minutes, Fouls, AST)
            card_content = html.Div([
                html.Div(stat["label"], style={"fontSize": "0.75rem", "fontWeight": "600", "color": "var(--text-secondary)", "marginBottom": "4px"}),
                html.Div(f"{stat['value']:.1f}", style={"fontSize": "1.1rem", "fontWeight": "700"}),
            ], id={"type": "supporting-stat-btn", "index": stat["key"]}, n_clicks=0, className=card_class)

        cards.append(card_content)

    return cards, selected_stat


# Callback to handle clicking on supporting stat cards
@callback(
    Output("selected-shooting-stat", "data", allow_duplicate=True),
    [Input({"type": "supporting-stat-btn", "index": ALL}, "n_clicks")],
    prevent_initial_call=True
)
def select_supporting_stat(clicks):
    from dash import ctx
    if not ctx.triggered_id:
        return "FG"
    return ctx.triggered_id["index"]


@callback(
    Output("shooting-breakdown-chart", "figure"),
    [Input("player-dropdown", "value"),
     Input("selected-shooting-stat", "data"),
     Input("selected-period", "data"),
     Input("selected-season", "data")]
)
def update_shooting_breakdown_chart(player_name, selected_stat, period, season):
    """Create bar chart showing stats per game. H2H does NOT affect this chart."""
    fig = go.Figure()

    if not player_name or not selected_stat:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            height=300
        )
        return fig

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # Filter by season or period (H2H does NOT affect supporting stats)
    if season:
        player_df = player_df[player_df["SEASON"] == season]
    elif period:
        player_df = player_df.head(period)

    # Cap at 20 games max for readability — fewer is better than crowded
    player_df = player_df.head(20)
    player_df = player_df.iloc[::-1]  # Reverse for chronological order

    if len(player_df) == 0:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            height=300
        )
        return fig

    n_games = len(player_df)

    # Build x-axis labels (date + opponent)
    labels = []
    for _, row in player_df.iterrows():
        date_str = row["_date"].strftime("%-m/%d") if pd.notna(row["_date"]) else ""
        matchup = row.get("MATCHUP", "")
        if isinstance(matchup, str):
            if "@" in matchup:
                opp = "@ " + matchup.split("@")[-1].strip()[:3]
            elif "vs." in matchup:
                opp = "vs " + matchup.split("vs.")[-1].strip()[:3]
            else:
                opp = ""
        else:
            opp = ""
        labels.append(f"{date_str}<br>{opp}")

    # Distinct colors per stat type
    _STAT_COLORS = {
        "FG":  ("#fbbf24", "#78350f"),   # amber made / dark missed
        "FG3": ("#60a5fa", "#1e3a5f"),   # blue made / dark missed
        "FT":  ("#34d399", "#064e3b"),   # green made / dark missed
        "MIN": "#94a3b8",                # slate
        "AST": "#f97066",                # coral
        "PF":  "#fb923c",                # orange
    }

    # Check if this is a shooting stat (stacked) or simple stat
    shooting_stats = {
        "FG": ("FGM", "FGA", "Field Goals"),
        "FG3": ("FG3M", "FG3A", "3-Pointers"),
        "FT": ("FTM", "FTA", "Free Throws")
    }

    simple_stats = {
        "MIN": ("MIN", "Minutes"),
        "AST": ("AST", "Assists"),
        "PF": ("PF", "Fouls")
    }

    if selected_stat in shooting_stats:
        made_col, attempts_col, title = shooting_stats[selected_stat]

        if made_col not in player_df.columns or attempts_col not in player_df.columns:
            return fig

        made_color, missed_color = _STAT_COLORS.get(selected_stat, ("#4a5568", "#718096"))
        made     = pd.to_numeric(player_df[made_col],     errors="coerce").fillna(0).astype(int)
        attempts = pd.to_numeric(player_df[attempts_col], errors="coerce").fillna(0).astype(int)
        missed   = (attempts - made).clip(lower=0)
        pct      = (made / attempts.replace(0, np.nan) * 100).fillna(0)

        fig.add_trace(go.Bar(
            x=list(range(n_games)), y=made, name="Made",
            marker_color=made_color,
            text=[f"{m}" for m in made],
            textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate="Made: %{y}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=list(range(n_games)), y=missed, name="Missed",
            marker_color=missed_color,
            text=[f"{int(p)}%" for p in pct],
            textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate="Missed: %{y}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_games)), y=attempts + 0.5,
            mode="text", text=[str(a) for a in attempts],
            textposition="top center",
            textfont=dict(size=11, color=COLORS["text"]),
            hoverinfo="skip", showlegend=False
        ))
        fig.update_layout(barmode="stack")

    elif selected_stat in simple_stats:
        col, title = simple_stats[selected_stat]

        if col not in player_df.columns:
            return fig

        # Coerce to numeric (handles "35:20" MIN strings → NaN → 0)
        values  = pd.to_numeric(player_df[col], errors="coerce").fillna(0)
        avg_val = values.mean()
        bar_color = _STAT_COLORS.get(selected_stat, "#64748b")

        # Above avg = stat color, below = dimmed version
        bar_colors = [bar_color if v >= avg_val else "rgba(100,116,139,0.45)" for v in values]
        fmt = ".1f" if col == "MIN" else ".0f"

        fig.add_trace(go.Bar(
            x=list(range(n_games)), y=values,
            marker_color=bar_colors,
            text=[f"{v:{fmt}}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate=f"{title}: %{{y:{fmt}}}<extra></extra>"
        ))
        fig.add_hline(
            y=avg_val, line_dash="dash",
            line_color=COLORS["text_secondary"], line_width=2,
            annotation_text=f"Avg {avg_val:{fmt}}",
            annotation_position="top left",
            annotation_font_color=COLORS["text_secondary"]
        )
    else:
        # Fallback to FG
        return update_shooting_breakdown_chart(player_name, "FG", period, season)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=40, r=20, t=40, b=70 if n_games > 10 else 50),
        height=320,
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_games)),
            ticktext=labels,
            tickfont=dict(size=9 if n_games > 12 else 10, color=COLORS["text_muted"]),
            tickangle=-40 if n_games > 10 else 0,
            showgrid=False,
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            tickfont=dict(size=10, color=COLORS["text_muted"]),
            showgrid=True,
        ),
        bargap=0.25,
    )

    return fig


@callback(
    Output("player-insights", "children"),
    [Input("player-dropdown", "value"),
     Input("selected-period", "data"),
     Input("selected-season", "data"),
     Input("selected-h2h", "data"),
     Input("selected-location", "data")]
)
def generate_player_insights(player_name, period, season, h2h_mode, location):
    """Generate smart insights about player performance trends"""
    if not player_name:
        return "Select a player to see insights."

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # H2H mode: filter by today's opponent (last 10 games vs today's opponent)
    h2h_opponent = ""
    if h2h_mode == "h2h":
        player_df, h2h_opponent = filter_h2h_games(player_df, h2h_mode, player_name)
        if len(player_df) < 2:
            return f"Not enough H2H games vs {h2h_opponent if h2h_opponent else 'opponent'} to generate insights."
    # Location filter: filter by home or away games
    elif location == "home":
        if "MATCHUP" in player_df.columns:
            player_df = player_df[player_df["MATCHUP"].str.contains("vs.", na=False)]
    elif location == "away":
        if "MATCHUP" in player_df.columns:
            player_df = player_df[player_df["MATCHUP"].str.contains("@", na=False)]
    elif season:
        player_df = player_df[player_df["SEASON"] == season]
    elif period:
        player_df = player_df.head(period)

    if len(player_df) < 5:
        return "Not enough games to generate insights."

    insights = []

    # Helper to calculate hit rate
    def hit_rate(df, col, line):
        if col not in df.columns or len(df) == 0:
            return 0, 0
        hits = (df[col] > line).sum()
        return hits, len(df)

    # 1. Home vs Road performance
    if "IS_HOME" in player_df.columns or "MATCHUP" in player_df.columns:
        # Determine home/away from MATCHUP if IS_HOME not available
        if "IS_HOME" not in player_df.columns:
            player_df["IS_HOME"] = player_df["MATCHUP"].apply(
                lambda x: 0 if isinstance(x, str) and "@" in x else 1
            )

        road_df = player_df[player_df["IS_HOME"] == 0]
        home_df = player_df[player_df["IS_HOME"] == 1]

        if len(road_df) >= 5 and len(home_df) >= 5:
            # Check PTS+AST on the road
            if "PTS" in player_df.columns and "AST" in player_df.columns:
                road_df = road_df.copy()
                road_df["PTS_AST"] = road_df["PTS"] + road_df["AST"]
                avg_combo = road_df["PTS_AST"].mean()
                line = round(avg_combo * 0.8, 1)  # Set line at 80% of average
                hits, total = hit_rate(road_df, "PTS_AST", line)

                if hits >= total * 0.7 and total >= 10:  # 70%+ hit rate with 10+ games
                    insights.append(
                        f"{player_name.split()[-1]} has exceeded {line} points + assists in "
                        f"{hits} of his last {total} games on the road ({avg_combo:.1f} points + assists/game average)."
                    )

    # 2. Recent streak analysis
    if "PTS" in player_df.columns:
        recent_15 = player_df.head(15)
        if len(recent_15) >= 10:
            pts_avg = recent_15["PTS"].mean()
            line = round(pts_avg * 0.85, 1)
            hits, total = hit_rate(recent_15, "PTS", line)

            if hits >= total * 0.8:
                insights.append(
                    f"Hit {line}+ points in {hits} of last {total} games ({pts_avg:.1f} PPG average)."
                )

    # 3. Rebounds consistency
    if "REB" in player_df.columns:
        recent = player_df.head(15)
        if len(recent) >= 10:
            reb_avg = recent["REB"].mean()
            line = round(reb_avg * 0.8, 1)
            hits, total = hit_rate(recent, "REB", line)

            if hits >= total * 0.75 and reb_avg >= 4:
                insights.append(
                    f"Grabbed {line}+ rebounds in {hits} of last {total} games ({reb_avg:.1f} RPG)."
                )

    # 4. PRA (Points + Rebounds + Assists) combo
    if all(col in player_df.columns for col in ["PTS", "REB", "AST"]):
        recent = player_df.head(15).copy()
        recent["PRA"] = recent["PTS"] + recent["REB"] + recent["AST"]
        pra_avg = recent["PRA"].mean()
        line = round(pra_avg * 0.85, 1)
        hits = (recent["PRA"] > line).sum()
        total = len(recent)

        if hits >= total * 0.7 and pra_avg >= 20:
            insights.append(
                f"Exceeded {line} PRA (Pts+Reb+Ast) in {hits} of last {total} games ({pra_avg:.1f} average)."
            )

    # 5. 3-pointer trends
    if "FG3M" in player_df.columns:
        recent = player_df.head(15)
        if len(recent) >= 10:
            avg_3pm = recent["FG3M"].mean()
            if avg_3pm >= 1.5:
                line = round(avg_3pm * 0.7, 1)
                hits, total = hit_rate(recent, "FG3M", line)

                if hits >= total * 0.75:
                    insights.append(
                        f"Made {line}+ three-pointers in {hits} of last {total} games ({avg_3pm:.1f} 3PM average)."
                    )

    # 6. Assists prop
    if "AST" in player_df.columns:
        recent = player_df.head(15)
        if len(recent) >= 10:
            ast_avg = recent["AST"].mean()
            if ast_avg >= 3:
                line = round(ast_avg * 0.8, 1)
                hits, total = hit_rate(recent, "AST", line)

                if hits >= total * 0.75:
                    insights.append(
                        f"Over {line} assists in {hits} of last {total} games ({ast_avg:.1f} APG average)."
                    )

    # 7. Points + Rebounds combo
    if "PTS" in player_df.columns and "REB" in player_df.columns:
        recent = player_df.head(15).copy()
        recent["PTS_REB"] = recent["PTS"] + recent["REB"]
        combo_avg = recent["PTS_REB"].mean()
        if combo_avg >= 15:
            line = round(combo_avg * 0.85, 1)
            hits = (recent["PTS_REB"] > line).sum()
            total = len(recent)

            if hits >= total * 0.7:
                insights.append(
                    f"Over {line} points + rebounds in {hits} of last {total} games ({combo_avg:.1f} P+R average)."
                )

    if not insights:
        # Generate a basic betting insight as fallback
        recent = player_df.head(10)
        if "PTS" in recent.columns and len(recent) >= 5:
            pts_avg = recent["PTS"].mean()
            line = round(pts_avg * 0.85, 1)
            hits = (recent["PTS"] > line).sum()
            insights.append(f"Over {line} points in {hits} of last {len(recent)} games ({pts_avg:.1f} PPG).")

    return " ".join(insights)


def create_matchup_content(player_name, stat):
    """Create the matchup analysis sidebar content matching the Outlier-style design"""
    player_df = DF[DF["PLAYER_NAME"] == player_name]

    if len(player_df) == 0:
        return html.Div("No data available", style={"color": COLORS["text_muted"]})

    # Get opponent team from most recent game
    last_game = player_df.sort_values("_date", ascending=False).iloc[0]
    matchup = last_game.get("MATCHUP", "")
    opponent = ""
    if isinstance(matchup, str):
        if "@" in matchup:
            opponent = matchup.split("@")[-1].strip()[:3]
        elif "vs." in matchup:
            opponent = matchup.split("vs.")[-1].strip()[:3]

    # Get player's position from PLAYER_POSITIONS
    player_position = "F"  # Default
    if not PLAYER_POSITIONS.empty:
        pos_match = PLAYER_POSITIONS[PLAYER_POSITIONS["PLAYER_NAME"] == player_name]
        if len(pos_match) > 0:
            player_position = pos_match["POSITION"].iloc[0]

    # Position display names
    position_display = {"G": "Guards", "F": "Forwards", "C": "Centers"}
    player_pos_display = position_display.get(player_position, "Forwards")

    # Position-specific defensive stats from DEFENSE_VS_POS
    pts_allowed = ast_allowed = reb_allowed = tpm_allowed = 0
    pts_rank = ast_rank = reb_rank = tpm_rank = 15

    if not DEFENSE_VS_POS.empty and opponent:
        pos_def = DEFENSE_VS_POS[
            (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opponent) &
            (DEFENSE_VS_POS["POSITION"] == player_position)
        ]
        if len(pos_def) > 0:
            row = pos_def.iloc[0]
            pts_allowed = row.get("AVG_PTS_ALLOWED", 0)
            ast_allowed = row.get("AVG_AST_ALLOWED", 0)
            reb_allowed = row.get("AVG_REB_ALLOWED", 0)
            tpm_allowed = row.get("AVG_3PM_ALLOWED", 0)
            pts_rank = int(row.get("PTS_RANK", 15))
            ast_rank = int(row.get("AST_RANK", 15))
            reb_rank = int(row.get("REB_RANK", 15))
            tpm_rank = int(row.get("3PM_RANK", 15))

    # Fallback to team-level defense stats if position-specific not available
    if pts_allowed == 0 and not TEAM_DEF.empty:
        opp_def = TEAM_DEF[TEAM_DEF["TEAM_ABBREVIATION"] == opponent] if "TEAM_ABBREVIATION" in TEAM_DEF.columns else pd.DataFrame()
        if not opp_def.empty:
            row = opp_def.iloc[0]
            pts_allowed = row.get("OPP_PTS", 0)
            ast_allowed = row.get("OPP_AST", 0)
            reb_allowed = row.get("OPP_REB", 0)
            tpm_allowed = row.get("OPP_FG3M", 0)
            pts_rank = int(row.get("OPP_PTS_RANK", 15))
            ast_rank = int(row.get("OPP_AST_RANK", 15))

    # Opponent W-L record and defensive context from TEAM_DEF
    opp_record = ""
    opp_fg_pct = opp_fg_rank = opp_tov = opp_tov_rank = None
    if not TEAM_DEF.empty and opponent:
        opp_row = TEAM_DEF[TEAM_DEF["TEAM_ABBREVIATION"] == opponent]
        if not opp_row.empty:
            r = opp_row.iloc[0]
            w, l = int(r.get("W", 0)), int(r.get("L", 0))
            opp_record = f"{w}-{l}"
            opp_fg_pct = r.get("OPP_FG_PCT", None)
            opp_fg_rank = r.get("OPP_FG_PCT_RANK", None)
            opp_tov = r.get("OPP_TOV", None)
            opp_tov_rank = r.get("OPP_TOV_RANK", None)

    # Position tabs - using actual position categories from data
    position_tabs = ["Overall", "vs Guards", "vs Forwards", "vs Centers"]
    current_pos_tab = f"vs {player_pos_display}"

    def get_rank_color(rank):
        if rank <= 10:
            return COLORS["pts"]  # Teal for high ranks (bad defense = good for player)
        elif rank <= 20:
            return COLORS["hit_mid"]
        else:
            return COLORS["text_secondary"]

    return html.Div([
        # Key Defense Section
        html.Div([
            # Header with W-L record
            html.Div([
                html.Span(f"Key {opponent} Defense vs {player_pos_display}", style={
                    "color": COLORS["text"], "fontSize": "14px", "fontWeight": "600"
                }),
                html.Span(f"  {opp_record}", style={
                    "color": COLORS["text_muted"], "fontSize": "12px", "marginLeft": "8px"
                }) if opp_record else None,
            ], style={"marginBottom": "8px"}),

            # Context line: Opp FG% + TOV forced
            html.Div([
                html.Span(
                    f"Opp FG% {opp_fg_pct*100:.1f}% (#{int(opp_fg_rank)})",
                    style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginRight": "12px"}
                ) if opp_fg_pct is not None and opp_fg_rank is not None else None,
                html.Span(
                    f"TOV forced {opp_tov:.1f}/g (#{int(opp_tov_rank)})",
                    style={"color": COLORS["text_secondary"], "fontSize": "11px"}
                ) if opp_tov is not None and opp_tov_rank is not None else None,
            ], style={"display": "flex", "marginBottom": "14px"}),

            # Position tabs
            html.Div([
                html.Span(
                    pos,
                    style={
                        "padding": "6px 12px",
                        "fontSize": "12px",
                        "color": COLORS["text"] if pos == current_pos_tab else COLORS["text_muted"],
                        "borderBottom": f"2px solid {COLORS['text']}" if pos == current_pos_tab else "none",
                        "cursor": "pointer",
                        "marginRight": "8px"
                    }
                ) for pos in position_tabs
            ], style={"display": "flex", "marginBottom": "16px", "flexWrap": "wrap"}),

            # Stats table — PTS / AST / REB / 3PM
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Stat (per game)", style={"textAlign": "left", "color": COLORS["text_muted"], "fontSize": "11px", "padding": "8px 0", "fontWeight": "400"}),
                        html.Th("Rank", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px", "fontWeight": "400"}),
                        html.Th("Avg", style={"textAlign": "right", "color": COLORS["text_muted"], "fontSize": "11px", "fontWeight": "400"}),
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Points Allowed", style={"padding": "9px 0", "fontSize": "13px", "fontWeight": "500"}),
                        html.Td(f"#{pts_rank}", style={"textAlign": "center", "color": get_rank_color(pts_rank), "fontSize": "14px", "fontWeight": "600"}),
                        html.Td(f"{pts_allowed:.1f}", style={"textAlign": "right", "fontSize": "13px"}),
                    ]),
                    html.Tr([
                        html.Td("Assists Allowed", style={"padding": "9px 0", "fontSize": "13px", "fontWeight": "500"}),
                        html.Td(f"#{ast_rank}", style={"textAlign": "center", "color": get_rank_color(ast_rank), "fontSize": "14px", "fontWeight": "600"}),
                        html.Td(f"{ast_allowed:.1f}", style={"textAlign": "right", "fontSize": "13px"}),
                    ]),
                    html.Tr([
                        html.Td("Rebounds Allowed", style={"padding": "9px 0", "fontSize": "13px", "fontWeight": "500"}),
                        html.Td(f"#{reb_rank}", style={"textAlign": "center", "color": get_rank_color(reb_rank), "fontSize": "14px", "fontWeight": "600"}),
                        html.Td(f"{reb_allowed:.1f}", style={"textAlign": "right", "fontSize": "13px"}),
                    ]),
                    html.Tr([
                        html.Td("3-Pointers Allowed", style={"padding": "9px 0", "fontSize": "13px", "fontWeight": "500"}),
                        html.Td(f"#{tpm_rank}", style={"textAlign": "center", "color": get_rank_color(tpm_rank), "fontSize": "14px", "fontWeight": "600"}),
                        html.Td(f"{tpm_allowed:.1f}", style={"textAlign": "right", "fontSize": "13px"}),
                    ]),
                ])
            ], style={"width": "100%", "borderCollapse": "collapse"})
        ], style=CARD),

        # Team Rankings Section
        html.Div([
            html.Div([
                html.Div(f"Team Rankings ({CURRENT_SEASON.split('-')[0]})", style={
                    "color": COLORS["text"],
                    "fontSize": "14px",
                    "fontWeight": "600",
                }),
            ], style={"marginBottom": "16px"}),

            # Offense/Defense toggle
            html.Div([
                html.Span("Offense", style={
                    "padding": "6px 16px",
                    "fontSize": "12px",
                    "backgroundColor": COLORS["border"],
                    "borderRadius": "4px",
                    "color": COLORS["text"],
                    "marginRight": "8px"
                }),
                html.Span("↔", style={"color": COLORS["text_muted"], "marginRight": "8px"}),
                html.Span("Defense", style={
                    "padding": "6px 16px",
                    "fontSize": "12px",
                    "color": COLORS["text_muted"],
                }),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

            # Team comparison table
            create_team_rankings_table(player_df, opponent)
        ], style=CARD),

        # Hit rates table
        html.Div([
            html.Div("HIT RATES", style={
                "color": COLORS["text_secondary"],
                "fontSize": "11px",
                "fontWeight": "600",
                "letterSpacing": "1px",
                "marginBottom": "12px"
            }),
            create_hit_rates_table(player_name)
        ], style=CARD),
    ])


def create_team_rankings_table(player_df, opponent):
    """Create team rankings comparison table.

    Left column: player's team offensive stats (from TEAM_OFF).
    Right column: opponent defensive stats (from TEAM_DEF).
    """
    # Get player's team from most recent game matchup
    last_game = player_df.sort_values("_date", ascending=False).iloc[0]
    matchup = last_game.get("MATCHUP", "")
    player_team = matchup.split()[0] if isinstance(matchup, str) and matchup else "UNK"

    # Player team → offensive stats (TEAM_OFF = team_stats.csv)
    player_off = TEAM_OFF[TEAM_OFF["TEAM_ABBREVIATION"] == player_team] if not TEAM_OFF.empty else pd.DataFrame()
    # Opponent → defensive stats (TEAM_DEF = team_defensive_stats.csv)
    opp_def = TEAM_DEF[TEAM_DEF["TEAM_ABBREVIATION"] == opponent] if "TEAM_ABBREVIATION" in TEAM_DEF.columns else pd.DataFrame()

    # (stat_label, off_col, off_rank_col, def_col, def_rank_col, is_pct)
    stats_to_show = [
        ("Points",   "PTS",     None,              "OPP_PTS",     "OPP_PTS_RANK",     False),
        ("FG%",      "FG_PCT",  None,              "OPP_FG_PCT",  "OPP_FG_PCT_RANK",  True),
        ("3P%",      "FG3_PCT", None,              "OPP_FG3_PCT", "OPP_FG3_PCT_RANK", True),
        ("Rebounds", "REB",     None,              "OPP_REB",     "OPP_REB_RANK",     False),
        ("Assists",  "AST",     None,              "OPP_AST",     "OPP_AST_RANK",     False),
    ]

    rows = []
    for stat_name, off_col, _, def_col, def_rank_col, is_pct in stats_to_show:
        # Player team offensive value + rank (rank within TEAM_OFF)
        pt_val = 0.0
        pt_rank = "-"
        if not player_off.empty and off_col in player_off.columns:
            pt_val = float(player_off.iloc[0].get(off_col, 0) or 0)
            if off_col in TEAM_OFF.columns:
                try:
                    pt_rank = int(TEAM_OFF[off_col].rank(ascending=False, method="min").loc[player_off.index[0]])
                except Exception:
                    pt_rank = "-"

        # Opponent defensive value + rank (use pre-computed rank column)
        opp_val = 0.0
        opp_rank = "-"
        if not opp_def.empty:
            opp_val = float(opp_def.iloc[0].get(def_col, 0) or 0)
            if def_rank_col in opp_def.columns:
                opp_rank = int(opp_def.iloc[0].get(def_rank_col, 15))

        # Format
        if is_pct:
            pt_display  = f"{pt_val * 100:.1f}%"  if pt_val  < 1 else f"{pt_val:.1f}%"
            opp_display = f"{opp_val * 100:.1f}%" if opp_val < 1 else f"{opp_val:.1f}%"
        else:
            pt_display  = f"{pt_val:.1f}"
            opp_display = f"{opp_val:.1f}"

        rows.append(html.Tr([
            html.Td(pt_display, style={"fontSize": "12px", "fontWeight": "500", "padding": "8px 0"}),
            html.Td(f"{pt_rank}", style={"fontSize": "11px", "color": COLORS["text_muted"]}),
            html.Td(stat_name, style={"fontSize": "12px", "textAlign": "center", "color": COLORS["text_secondary"]}),
            html.Td(f"{opp_rank}", style={"fontSize": "11px", "color": COLORS["text_muted"], "textAlign": "right"}),
            html.Td(opp_display, style={"fontSize": "12px", "fontWeight": "500", "textAlign": "right", "padding": "8px 0"}),
        ]))

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Avg.", style={"textAlign": "left", "color": COLORS["text_muted"], "fontSize": "10px", "fontWeight": "400"}),
                html.Th("Rank", style={"color": COLORS["text_muted"], "fontSize": "10px", "fontWeight": "400"}),
                html.Th("Stat", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "10px", "fontWeight": "400"}),
                html.Th("Rank", style={"textAlign": "right", "color": COLORS["text_muted"], "fontSize": "10px", "fontWeight": "400"}),
                html.Th("Avg.", style={"textAlign": "right", "color": COLORS["text_muted"], "fontSize": "10px", "fontWeight": "400"}),
            ])
        ]),
        html.Tbody(rows)
    ], style={"width": "100%", "borderCollapse": "collapse"})


def create_injuries_content(player_name):
    """Create the injuries tab content with clickable news links"""
    try:
        status = get_player_injury_status(player_name)
        status_text = status.get("status", "ACTIVE")
        reason = status.get("reason", "")
        news = status.get("news", [])
        injury_type = status.get("injury_type", "")

        if status_text == "HEALTHY":
            status_text = "ACTIVE"
        # UNKNOWN stays UNKNOWN — don't pretend player is active when we don't know
    except Exception:
        status_text = "UNKNOWN"
        reason = ""
        news = []
        injury_type = ""

    # Status colors and labels
    status_config = {
        "ACTIVE":       {"color": "#22c55e",  "label": "ACTIVE",       "icon": "✓"},
        "PROBABLE":     {"color": "#a3e635",  "label": "PROBABLE",     "icon": "●"},
        "QUESTIONABLE": {"color": "#f59e0b",  "label": "QUESTIONABLE", "icon": "?"},
        "DOUBTFUL":     {"color": "#f97316",  "label": "DOUBTFUL",     "icon": "!"},
        "OUT":          {"color": "#f43f5e",  "label": "OUT",          "icon": "✕"},
        "UNKNOWN":      {"color": "#64748b",  "label": "STATUS UNAVAILABLE", "icon": "—"},
    }
    config = status_config.get(status_text, status_config["ACTIVE"])

    player_df = DF[DF["PLAYER_NAME"] == player_name]
    games_current = len(player_df[player_df["SEASON"] == CURRENT_SEASON])

    # Build news items with clickable links
    news_elements = []
    for n in (news[:5] if news else []):
        title = n.get("title", "")
        link = n.get("link", "")
        source = n.get("source", "Unknown")

        if title:
            news_elements.append(
                html.Div([
                    # Clickable title - opens in new tab
                    html.A(
                        title,
                        href=link,
                        target="_blank",  # Opens in new tab
                        rel="noopener noreferrer",
                        style={
                            "fontSize": "13px",
                            "color": COLORS["text"],
                            "textDecoration": "none",
                            "cursor": "pointer",
                            "display": "block",
                            "marginBottom": "6px",
                            ":hover": {"textDecoration": "underline"}
                        }
                    ) if link else html.Div(title, style={"fontSize": "13px", "marginBottom": "6px"}),
                    # Source with link
                    html.Div([
                        html.A(
                            f"📰 {source}",
                            href=link,
                            target="_blank",
                            rel="noopener noreferrer",
                            style={
                                "fontSize": "11px",
                                "color": COLORS["pts"],
                                "textDecoration": "none"
                            }
                        ) if link else html.Span(source, style={"fontSize": "11px", "color": COLORS["text_muted"]}),
                    ])
                ], style={
                    "marginBottom": "16px",
                    "paddingBottom": "16px",
                    "borderBottom": f"1px solid {COLORS['border']}"
                })
            )

    if not news_elements:
        news_elements = [html.Div("No recent injury news", style={"color": COLORS["text_muted"], "fontSize": "13px"})]

    return html.Div([
        # Status card
        html.Div([
            html.Div("INJURY STATUS", style={
                "color": COLORS["text_secondary"],
                "fontSize": "11px",
                "fontWeight": "600",
                "letterSpacing": "1px",
                "marginBottom": "16px"
            }),
            html.Div([
                html.Span(config["icon"], style={
                    "color": config["color"],
                    "marginRight": "12px",
                    "fontSize": "20px",
                    "fontWeight": "bold"
                }),
                html.Span(config["label"], style={
                    "color": config["color"],
                    "fontWeight": "700",
                    "fontSize": "20px"
                })
            ]),
            html.Div(reason, style={
                "color": COLORS["text_secondary"],
                "fontSize": "13px",
                "marginTop": "10px"
            }) if reason and reason != "No injury news found" else None,
            html.Div(f"Injury: {injury_type.title()}", style={
                "color": COLORS["text_muted"],
                "fontSize": "12px",
                "marginTop": "8px"
            }) if injury_type else None,
            html.Div(f"{games_current} games played in {CURRENT_SEASON}", style={
                "color": COLORS["text_muted"],
                "fontSize": "12px",
                "marginTop": "12px"
            }),
        ], style=CARD),

        # Recent news with clickable links
        html.Div([
            html.Div([
                html.Span("RECENT NEWS", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "letterSpacing": "1px",
                }),
                html.Span(" (click to view source)", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "10px",
                    "marginLeft": "8px"
                })
            ], style={"marginBottom": "16px"}),
        ], style=CARD),
    ])


def create_insights_content(player_name, _stat=None):
    """Create the AI Expert Insight panel with deep analysis including matchup data"""
    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    if len(player_df) < 5:
        return html.Div("Not enough data for analysis.", style={"color": COLORS["text_muted"], "padding": "20px"})

    # Calculate key stats
    l5 = player_df.head(5)
    l10 = player_df.head(10)
    season_avg = player_df["PTS"].mean()
    l5_avg = l5["PTS"].mean()
    l5_reb = l5["REB"].mean() if "REB" in l5.columns else 0
    l5_ast = l5["AST"].mean() if "AST" in l5.columns else 0

    # Get player's team and position
    player_team = player_df.iloc[0].get("TEAM_ABBREVIATION", "his team")

    # Get player position
    player_position = "G"  # Default
    if not PLAYER_POSITIONS.empty:
        pos_match = PLAYER_POSITIONS[PLAYER_POSITIONS["PLAYER_NAME"] == player_name]
        if len(pos_match) > 0:
            pos = pos_match["POSITION"].iloc[0]
            if "G" in str(pos):
                player_position = "G"
            elif "F" in str(pos):
                player_position = "F"
            elif "C" in str(pos):
                player_position = "C"

    # Get today's opponent from teams playing today or last matchup
    teams_today = get_teams_playing_today()
    opponent = ""
    opponent_full = ""

    # Check if player's team is playing today
    if player_team in teams_today:
        # Find opponent from today's games
        today_games = get_todays_games()
        for _, game in today_games.iterrows():
            home = game.get("HOME_TEAM", "")
            away = game.get("AWAY_TEAM", "")
            if player_team == home:
                opponent = away
                break
            elif player_team == away:
                opponent = home
                break

    # If no game today, get next upcoming opponent
    if not opponent:
        next_opp, next_date = get_next_opponent_for_team(player_team)
        if next_opp:
            opponent = next_opp

    # Team name mapping for display
    team_names = {
        "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
        "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
        "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
        "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
        "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
    }
    opponent_full = team_names.get(opponent, opponent)

    # Get opponent's defensive ranking vs player's position
    opp_def_rank_pts = 15  # Default middle
    opp_def_rank_ast = 15
    opp_def_rank_reb = 15
    opp_pts_allowed = l5_avg  # Default
    opp_ast_allowed = l5_ast
    matchup_analysis = ""

    if not DEFENSE_VS_POS.empty and opponent:
        # Get current season defense data
        opp_def = DEFENSE_VS_POS[
            (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opponent) &
            (DEFENSE_VS_POS["POSITION"] == player_position) &
            (DEFENSE_VS_POS["SEASON"] == CURRENT_SEASON)
        ]
        if len(opp_def) == 0:
            # Try previous season
            opp_def = DEFENSE_VS_POS[
                (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opponent) &
                (DEFENSE_VS_POS["POSITION"] == player_position)
            ].sort_values("SEASON", ascending=False).head(1)

        if len(opp_def) > 0:
            opp_def_rank_pts = int(opp_def["PTS_RANK"].iloc[0])
            opp_def_rank_ast = int(opp_def["AST_RANK"].iloc[0])
            opp_def_rank_reb = int(opp_def["REB_RANK"].iloc[0])
            opp_pts_allowed = opp_def["AVG_PTS_ALLOWED"].iloc[0]
            opp_ast_allowed = opp_def["AVG_AST_ALLOWED"].iloc[0]

    # Categorize defense strength
    def_strength_pts = "elite" if opp_def_rank_pts <= 6 else "good" if opp_def_rank_pts <= 15 else "weak"
    def_strength_ast = "elite" if opp_def_rank_ast <= 6 else "good" if opp_def_rank_ast <= 15 else "weak"

    pos_name = {"G": "guards", "F": "forwards", "C": "centers"}.get(player_position, "players")

    # Calculate performance vs similar defenses
    # Look at games vs teams with similar defensive rank
    similar_def_games = []
    for _, row in l10.iterrows():
        matchup = row.get("MATCHUP", "")
        if isinstance(matchup, str):
            if "@" in matchup:
                game_opp = matchup.split("@")[-1].strip()[:3]
            elif "vs." in matchup:
                game_opp = matchup.split("vs.")[-1].strip()[:3]
            else:
                continue

            # Check this opponent's defensive rank
            game_opp_def = DEFENSE_VS_POS[
                (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == game_opp) &
                (DEFENSE_VS_POS["POSITION"] == player_position)
            ]
            if len(game_opp_def) > 0:
                game_opp_rank = game_opp_def["PTS_RANK"].iloc[0]
                # Similar defense = within 5 ranks
                if abs(game_opp_rank - opp_def_rank_pts) <= 8:
                    similar_def_games.append(row)

    # Calculate hit rate vs similar defenses
    similar_hits = 0
    similar_total = len(similar_def_games)
    similar_avg_pts = l5_avg
    similar_avg_ast = l5_ast

    if similar_total >= 3:
        similar_df = pd.DataFrame(similar_def_games)
        similar_avg_pts = similar_df["PTS"].mean()
        similar_avg_ast = similar_df["AST"].mean() if "AST" in similar_df.columns else 0

    # Determine the line (use ~85% of average as typical line)
    pts_line = round(l5_avg * 0.9, 1)
    hits_l5 = (l5["PTS"] > pts_line).sum()
    hit_pct = int(hits_l5 / 5 * 100)

    # Calculate combo lines
    pa_avg = l5_avg + l5_ast  # Points + Assists
    pa_line = round(pa_avg * 0.9, 1)
    pa_hits = ((l5["PTS"] + l5["AST"]) > pa_line).sum() if "AST" in l5.columns else 0

    # Find recent high games
    recent_highs = l5[l5["PTS"] >= 30]
    high_game_opponents = []
    for _, row in recent_highs.iterrows():
        matchup = row.get("MATCHUP", "")
        if isinstance(matchup, str):
            if "@" in matchup:
                opp = matchup.split("@")[-1].strip()[:3]
            elif "vs." in matchup:
                opp = matchup.split("vs.")[-1].strip()[:3]
            else:
                opp = "OPP"
            high_game_opponents.append(opp)

    # Find the worst recent game
    worst_game = l5.loc[l5["PTS"].idxmin()]
    worst_pts = worst_game["PTS"]
    worst_matchup = worst_game.get("MATCHUP", "")
    worst_opp = ""
    if isinstance(worst_matchup, str):
        if "@" in worst_matchup:
            worst_opp = worst_matchup.split("@")[-1].strip()[:3]
        elif "vs." in worst_matchup:
            worst_opp = worst_matchup.split("vs.")[-1].strip()[:3]

    # Build bullish indicators with matchup context
    bullish = []

    # Matchup-based bullish
    if def_strength_pts == "weak":
        bullish.append(f"Favorable matchup: {opponent_full} ranks #{opp_def_rank_pts} defending {pos_name} (bottom 10).")
    if def_strength_ast == "weak" and l5_ast >= 4:
        bullish.append(f"{opponent_full} allows {opp_ast_allowed:.1f} assists to {pos_name} (rank #{opp_def_rank_ast}).")

    if hit_pct >= 80:
        bullish.append(f"Hit the over in {hit_pct}% of the last five games.")
    elif hit_pct >= 60:
        bullish.append(f"Hit the over in {hit_pct}% of recent games.")

    if len(high_game_opponents) >= 2:
        bullish.append(f"Back-to-back 30+ point games against {' and '.join(high_game_opponents)}.")
    elif len(high_game_opponents) == 1:
        bullish.append(f"Recorded a 30+ point game against {high_game_opponents[0]} recently.")

    if l5_avg >= 20:
        bullish.append(f"Primary scoring option for {player_team} ({l5_avg:.1f} PPG over L5).")

    if l5_ast >= 5:
        bullish.append(f"Strong playmaking: {l5_ast:.1f} assists over last 5 games.")

    if l5_reb >= 8:
        bullish.append(f"Dominant on the glass: {l5_reb:.1f} rebounds per game.")

    # Build risk factors with matchup context
    risks = []

    # Matchup-based risks
    if def_strength_pts == "elite":
        risks.append(f"Tough matchup: {opponent_full} ranks #{opp_def_rank_pts} defending {pos_name} (top 6).")

    if worst_pts < pts_line * 0.7:
        risks.append(f"Dropped to {int(worst_pts)} points against {worst_opp} recently.")

    pts_std = l5["PTS"].std()
    if pts_std > 8:
        risks.append("High game-to-game variance in scoring output.")

    if "MIN" in l5.columns:
        min_trend = l5["MIN"].iloc[0] - l5["MIN"].mean()
        if min_trend < -3:
            risks.append("Recent decrease in playing time may impact production.")

    if not risks:
        risks.append("No significant risk factors identified.")

    # Build matchup-aware recommendation
    # Get model prediction (lazy loaded)
    model_prediction = l5_avg
    pts_predictor = get_predictor("PTS")
    if pts_predictor:
        try:
            result = pts_predictor.predict_player_game(player_name, DF)
            if "error" not in result:
                model_prediction = result.get("predicted_pts", l5_avg)
        except:
            pass

    # Adjust prediction based on matchup
    if def_strength_pts == "weak":
        matchup_adj = 1.05  # 5% boost vs weak defense
    elif def_strength_pts == "elite":
        matchup_adj = 0.92  # 8% reduction vs elite defense
    else:
        matchup_adj = 1.0

    adjusted_projection = model_prediction * matchup_adj

    # Build deep intelligence text with matchup
    if opponent:
        if def_strength_pts == "weak":
            matchup_text = (
                f"He is facing the {opponent_full} defense which ranks #{opp_def_rank_pts} against {pos_name}. "
                f"The model projects him to score {adjusted_projection:.1f} points. "
            )
            if similar_total >= 3:
                matchup_text += f"Against teams with similar weak defenses, he averages {similar_avg_pts:.1f} points."
        elif def_strength_pts == "elite":
            matchup_text = (
                f"Caution: He faces the {opponent_full} who rank #{opp_def_rank_pts} defending {pos_name}. "
                f"The model projects {adjusted_projection:.1f} points with this tough matchup factored in."
            )
        else:
            matchup_text = (
                f"He faces {opponent_full} (#{opp_def_rank_pts} vs {pos_name}). "
                f"Model projection: {adjusted_projection:.1f} points."
            )
    else:
        matchup_text = f"Model projection: {adjusted_projection:.1f} points based on recent form."

    # Final recommendation with matchup context
    if def_strength_pts == "weak" and hit_pct >= 60:
        recommendation = (
            f"Strong play: Over {pts_line} points. Favorable matchup vs {opponent} (#{opp_def_rank_pts} defense) "
            f"combined with {hit_pct}% hit rate makes this a high-confidence pick."
        )
        if l5_ast >= 4:
            recommendation += f" Also consider Over {pa_line} P+A ({pa_hits}/5 recent hits)."
    elif def_strength_pts == "elite":
        recommendation = (
            f"Proceed with caution on Over {pts_line}. The {opponent_full}'s elite defense (#{opp_def_rank_pts}) "
            f"could limit scoring. Consider the under or a lower line."
        )
    elif hit_pct >= 80:
        recommendation = f"Over {pts_line} points is statistically favored with {hit_pct}% hit rate."
    elif hit_pct >= 60:
        recommendation = f"Moderate edge on Over {pts_line} ({hit_pct}% hit rate). Matchup is neutral."
    else:
        recommendation = f"Exercise caution on Over {pts_line}. Recent hit rate of {hit_pct}% suggests variance."

    # Build the UI
    return html.Div([
        # AI Expert Insight Header
        html.Div([
            html.Div([
                html.Span("⚡", style={"fontSize": "28px", "marginRight": "14px"}),
                html.Span("AI EXPERT INSIGHT", style={
                    "color": COLORS["text"],
                    "fontSize": "16px",
                    "fontWeight": "800",
                    "letterSpacing": "2.5px"
                })
            ], style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "padding": "24px",
                "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
                "borderRadius": "16px 16px 0 0"
            })
        ]),

        # Deep Intelligence Report with Matchup Analysis
        html.Div([
            html.Div([
                html.Span("⚡", style={"color": COLORS["accent"], "marginRight": "10px", "fontSize": "16px"}),
                html.Span("DEEP INTELLIGENCE REPORT", style={
                    "color": COLORS["accent"],
                    "fontSize": "13px",
                    "fontWeight": "700",
                    "letterSpacing": "1.5px"
                })
            ], style={"marginBottom": "18px"}),

            html.P(
                f"{player_name} has a {l5_avg:.1f} point average over his last 5 games, "
                f"clearing the {pts_line} line in {hits_l5} of 5 contests. {matchup_text}",
                style={
                    "color": COLORS["text"],
                    "fontSize": "15px",
                    "fontWeight": "500",
                    "lineHeight": "1.8",
                    "margin": "0"
                }
            )
        ], style={**CARD, "marginTop": "0", "borderRadius": "0", "padding": "24px"}),

        # Bullish Indicators
        html.Div([
            html.Div([
                html.Span("☆", style={"color": "#10b981", "marginRight": "10px", "fontSize": "18px"}),
                html.Span("BULLISH INDICATORS", style={
                    "color": "#10b981",
                    "fontSize": "13px",
                    "fontWeight": "700",
                    "letterSpacing": "1.5px"
                })
            ], style={"marginBottom": "18px"}),

            html.Ul([
                html.Li(indicator, style={
                    "color": COLORS["text"],
                    "fontSize": "14px",
                    "fontWeight": "500",
                    "lineHeight": "1.7",
                    "marginBottom": "10px",
                    "listStyleType": "disc",
                    "marginLeft": "18px"
                }) for indicator in bullish
            ] if bullish else [
                html.Li("Stable recent performance with consistent output.", style={
                    "color": COLORS["text"],
                    "fontSize": "14px",
                    "fontWeight": "500",
                    "marginLeft": "18px"
                })
            ], style={"margin": "0", "padding": "0"})
        ], style={
            "backgroundColor": "rgba(16, 185, 129, 0.1)",
            "padding": "20px 24px",
            "margin": "12px 0",
            "borderRadius": "12px",
            "borderLeft": "3px solid #10b981"
        }),

        # Risk Factors
        html.Div([
            html.Div([
                html.Span("⚠", style={"color": "#ef4444", "marginRight": "10px", "fontSize": "18px"}),
                html.Span("RISK FACTORS", style={
                    "color": "#ef4444",
                    "fontSize": "13px",
                    "fontWeight": "700",
                    "letterSpacing": "1.5px"
                })
            ], style={"marginBottom": "18px"}),

            html.Ul([
                html.Li(risk, style={
                    "color": COLORS["text"],
                    "fontSize": "14px",
                    "fontWeight": "500",
                    "lineHeight": "1.7",
                    "marginBottom": "10px",
                    "listStyleType": "disc",
                    "marginLeft": "18px"
                }) for risk in risks
            ], style={"margin": "0", "padding": "0"})
        ], style={
            "backgroundColor": "rgba(239, 68, 68, 0.1)",
            "padding": "20px 24px",
            "margin": "12px 0",
            "borderRadius": "12px",
            "borderLeft": "4px solid #ef4444"
        }),

        # Final Recommendation
        html.Div([
            html.Div("FINAL RECOMMENDATION", style={
                "color": COLORS["text_secondary"],
                "fontSize": "13px",
                "fontWeight": "700",
                "letterSpacing": "1.5px",
                "marginBottom": "14px"
            }),

            html.Div(
                recommendation,
                style={
                    "backgroundColor": COLORS["bg"],
                    "padding": "20px",
                    "borderRadius": "12px",
                    "color": COLORS["text"],
                    "fontSize": "15px",
                    "fontWeight": "500",
                    "lineHeight": "1.8",
                    "borderLeft": f"4px solid {COLORS['accent']}"
                }
            )
        ], style={
            "borderTop": f"1px solid {COLORS['border']}",
            "paddingTop": "20px",
            "marginTop": "12px"
        })
    ], style={**CARD, "padding": "0", "overflow": "hidden"})


def create_best_props_content(_stat=None):
    """
    Create the Best Props tab content (reads from pre-computed cache).
    Shows WHY each prop was selected with matchup analysis.
    """
    from utils.props_cache import get_cached_props

    cache = get_cached_props()
    final_props = cache["sidebar_data"]
    teams_today = cache.get("teams_today", set())

    if not final_props:
        return html.Div([
            html.Div([
                html.Div("BEST PROPS", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "letterSpacing": "1px",
                    "marginBottom": "16px"
                }),
                html.Div("No favorable props found - check if models are trained", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "14px",
                    "textAlign": "center",
                    "padding": "40px 0"
                }),
            ], style=CARD)
        ])

    # Build prop cards with smart analysis
    prop_cards = []
    for prop in final_props:
        prob_color = get_hit_color(prop["hit_prob"] * 100)

        # Role badge styling
        role = prop.get("role", "UNKNOWN")
        role_colors = {
            "STARTER": {"bg": "#1a472a", "text": "#4ade80"},
            "ROTATION": {"bg": "#3d3d00", "text": "#facc15"},
            "BENCH": {"bg": "#4a1c1c", "text": "#f87171"},
            "UNKNOWN": {"bg": "#333", "text": "#888"}
        }
        role_style = role_colors.get(role, role_colors["UNKNOWN"])

        # Confidence badge
        confidence = prop.get("confidence", "LOW")
        conf_colors = {
            "HIGH": {"bg": "#1a472a", "text": "#4ade80"},
            "MEDIUM": {"bg": "#3d3d00", "text": "#facc15"},
            "LOW": {"bg": "#4a1c1c", "text": "#f87171"}
        }
        conf_style = conf_colors.get(confidence, conf_colors["LOW"])

        # Build positive factors (green)
        positive_elements = []
        for factor in prop.get("positive_factors", [])[:2]:
            positive_elements.append(
                html.Div([
                    html.Span("+ ", style={"color": "#4ade80", "fontWeight": "600"}),
                    html.Span(factor["reason"])
                ], style={
                    "fontSize": "11px",
                    "color": "#4ade80",
                    "marginBottom": "3px",
                    "paddingLeft": "8px",
                    "borderLeft": "2px solid #4ade80"
                })
            )

        # Build negative factors (orange/red)
        negative_elements = []
        for factor in prop.get("negative_factors", [])[:2]:
            negative_elements.append(
                html.Div([
                    html.Span("- " if factor["type"] != "role" else "! ", style={"color": "#f97316", "fontWeight": "600"}),
                    html.Span(factor["reason"])
                ], style={
                    "fontSize": "11px",
                    "color": "#f97316",
                    "marginBottom": "3px",
                    "paddingLeft": "8px",
                    "borderLeft": "2px solid #f97316"
                })
            )

        # Build opponent span
        opponent_span = html.Span(f" vs {prop['opponent']}", style={
            "color": COLORS["text_muted"],
            "fontSize": "12px",
            "marginLeft": "4px"
        }) if prop.get("opponent") else None

        prop_card = html.Div([
            # Header row: Player info + Probability
            html.Div([
                # Left: Player + Prop info
                html.Div([
                    html.Div([
                        html.Span(prop["player"], style={
                            "fontWeight": "600",
                            "fontSize": "14px"
                        }),
                        opponent_span,
                        html.Span(" " if prop.get("is_home") else " ", style={
                            "fontSize": "12px",
                            "marginLeft": "4px"
                        }),
                        # Role badge
                        html.Span(role, style={
                            "fontSize": "9px",
                            "fontWeight": "600",
                            "backgroundColor": role_style["bg"],
                            "color": role_style["text"],
                            "padding": "2px 6px",
                            "borderRadius": "4px",
                            "marginLeft": "6px"
                        }),
                    ]),
                    html.Div([
                        html.Span(f"O {prop['line']:.1f} ", style={
                            "color": COLORS["accent"],
                            "fontWeight": "600",
                            "fontSize": "14px"
                        }),
                        html.Span(prop["prop_label"], style={
                            "color": COLORS["text_secondary"],
                            "fontSize": "12px"
                        }),
                        html.Span(f" ({prop.get('avg_minutes', 0):.0f} min avg)", style={
                            "color": COLORS["text_muted"],
                            "fontSize": "11px",
                            "marginLeft": "6px"
                        }) if prop.get("avg_minutes", 0) > 0 else None
                    ], style={"marginTop": "2px"})
                ], style={"flex": "1"}),

                # Right: Probability + Confidence
                html.Div([
                    html.Div(f"{prop['hit_prob']*100:.0f}%", style={
                        "color": prob_color,
                        "fontWeight": "700",
                        "fontSize": "18px"
                    }),
                    html.Div([
                        html.Span(confidence, style={
                            "fontSize": "9px",
                            "fontWeight": "600",
                            "backgroundColor": conf_style["bg"],
                            "color": conf_style["text"],
                            "padding": "2px 6px",
                            "borderRadius": "4px"
                        })
                    ])
                ], style={"textAlign": "right"})
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "8px"
            }),

            # Smart Analysis section - positive and negative factors
            html.Div([
                *positive_elements,
                *negative_elements
            ], style={"marginTop": "8px"}) if (positive_elements or negative_elements) else None

        ], style={
            "padding": "14px",
            "backgroundColor": COLORS["card"],
            "borderRadius": "8px",
            "marginBottom": "12px",
            "border": f"1px solid {COLORS['border']}"
        })

        prop_cards.append(prop_card)

    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div("BEST PROPS", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "letterSpacing": "1px",
                }),
                html.Div(f"{len(teams_today)} games today • {len(final_props)} top picks", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "11px",
                    "marginTop": "4px",
                }),
            ]),
            html.Div([
                html.Span("Singles", style={
                    "padding": "4px 10px",
                    "fontSize": "11px",
                    "backgroundColor": COLORS["border"],
                    "borderRadius": "4px",
                    "marginRight": "6px"
                }),
                html.Span("Combos", style={
                    "padding": "4px 10px",
                    "fontSize": "11px",
                    "backgroundColor": COLORS["border"],
                    "borderRadius": "4px"
                }),
            ], style={"marginTop": "8px"})
        ], className="card"),

        # Prop cards with explanations
        html.Div(prop_cards),

        # Last updated
        html.Div([
            html.Div(f"Updated: {datetime.now().strftime('%I:%M %p')} • Refreshes every 5 min", style={
                "color": COLORS["text_muted"],
                "fontSize": "10px",
                "textAlign": "center"
            })
        ], style={"marginTop": "8px"})
    ])


def create_splits_table(player_name):
    """Create home/away splits table"""
    player_df = DF[DF["PLAYER_NAME"] == player_name]

    home = player_df[player_df["is_home"] == 1]
    away = player_df[player_df["is_home"] == 0]

    def get_avgs(df):
        if len(df) == 0:
            return 0, 0, 0
        return df["PTS"].mean(), df["AST"].mean(), df["REB"].mean()

    home_pts, home_ast, home_reb = get_avgs(home)
    away_pts, away_ast, away_reb = get_avgs(away)

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("", style={"textAlign": "left"}),
                html.Th("PTS", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
                html.Th("AST", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
                html.Th("REB", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td("Home", style={"padding": "6px 0", "fontSize": "12px"}),
                html.Td(f"{home_pts:.1f}", style={"textAlign": "center", "color": COLORS["pts"], "fontWeight": "500", "fontSize": "12px"}),
                html.Td(f"{home_ast:.1f}", style={"textAlign": "center", "color": COLORS["ast"], "fontWeight": "500", "fontSize": "12px"}),
                html.Td(f"{home_reb:.1f}", style={"textAlign": "center", "color": COLORS["reb"], "fontWeight": "500", "fontSize": "12px"}),
            ]),
            html.Tr([
                html.Td("Away", style={"padding": "6px 0", "fontSize": "12px"}),
                html.Td(f"{away_pts:.1f}", style={"textAlign": "center", "color": COLORS["pts"], "fontWeight": "500", "fontSize": "12px"}),
                html.Td(f"{away_ast:.1f}", style={"textAlign": "center", "color": COLORS["ast"], "fontWeight": "500", "fontSize": "12px"}),
                html.Td(f"{away_reb:.1f}", style={"textAlign": "center", "color": COLORS["reb"], "fontWeight": "500", "fontSize": "12px"}),
            ]),
        ])
    ], style={"width": "100%", "borderCollapse": "collapse"})


def create_hit_rates_table(player_name):
    """Create hit rates table for different stat lines"""
    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    def calc_hit(stat, line, n):
        recent = player_df.head(n)
        if len(recent) == 0:
            return 0
        if "+" in stat:
            parts = stat.split("+")
            vals = sum(recent[p] for p in parts if p in recent.columns)
        else:
            vals = recent[stat] if stat in recent.columns else pd.Series([0])
        return ((vals >= line).sum() / len(recent)) * 100

    # Calculate averages
    avgs = {
        "PTS": player_df["PTS"].mean(),
        "AST": player_df["AST"].mean(),
        "REB": player_df["REB"].mean(),
    }

    rows = []
    for stat, avg in avgs.items():
        line = round(avg * 2) / 2  # Round to nearest 0.5
        l5 = calc_hit(stat, line, 5)
        l10 = calc_hit(stat, line, 10)
        season = calc_hit(stat, line, len(player_df))

        rows.append(html.Tr([
            html.Td(f"{stat} O{line}", style={"padding": "6px 0", "fontSize": "12px"}),
            html.Td(f"{l5:.0f}%", style={"color": get_hit_color(l5), "textAlign": "center", "fontWeight": "500", "fontSize": "12px"}),
            html.Td(f"{l10:.0f}%", style={"color": get_hit_color(l10), "textAlign": "center", "fontWeight": "500", "fontSize": "12px"}),
            html.Td(f"{season:.0f}%", style={"color": get_hit_color(season), "textAlign": "center", "fontWeight": "500", "fontSize": "12px"}),
        ]))

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Line", style={"textAlign": "left", "color": COLORS["text_muted"], "fontSize": "11px", "padding": "6px 0"}),
                html.Th("L5", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
                html.Th("L10", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
                html.Th("Season", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
            ])
        ]),
        html.Tbody(rows)
    ], style={"width": "100%", "borderCollapse": "collapse"})


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("NBA Props Dashboard")
    print("=" * 50)
    print("\nOpen: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop\n")

    app.run(debug=False, port=8050)
