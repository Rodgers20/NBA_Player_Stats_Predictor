# dashboard/app.py
"""
NBA Player Props Dashboard 
=============================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from dash import Dash, html, dcc, Input, Output, callback, ALL
import plotly.graph_objects as go


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
from utils.injury_news import get_player_injury_status
from utils.prop_calculator import generate_best_props, calculate_hit_probability
from utils.data_fetch import get_todays_games, get_teams_playing_today
from models.predictor import NBAPredictor

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    game_logs = pd.read_csv(os.path.join(data_dir, "player_game_logs.csv"))
    team_def = pd.read_csv(os.path.join(data_dir, "team_defensive_stats.csv"))

    # Load optional position data
    positions_path = os.path.join(data_dir, "player_positions.csv")
    positions_df = pd.read_csv(positions_path) if os.path.exists(positions_path) else pd.DataFrame()

    # Load optional defense vs position data
    def_vs_pos_path = os.path.join(data_dir, "defense_vs_position.csv")
    def_vs_pos = pd.read_csv(def_vs_pos_path) if os.path.exists(def_vs_pos_path) else pd.DataFrame()

    # Load team stats for pace
    team_stats_path = os.path.join(data_dir, "team_stats.csv")
    team_stats = pd.read_csv(team_stats_path) if os.path.exists(team_stats_path) else pd.DataFrame()

    # Engineer features with all available data
    df = engineer_features(
        game_logs,
        team_def,
        team_stats=team_stats if not team_stats.empty else None,
        player_positions=positions_df if not positions_df.empty else None,
        defense_vs_position=def_vs_pos if not def_vs_pos.empty else None
    )
    df["_date"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df = df.sort_values(["PLAYER_NAME", "_date"])

    return df, team_def, positions_df, def_vs_pos


def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    predictors = {}
    for target in ["pts", "ast", "reb"]:
        filepath = os.path.join(models_dir, f"{target}_predictor.pkl")
        if os.path.exists(filepath):
            predictors[target.upper()] = NBAPredictor.load(filepath)
    return predictors


print("Loading data...")
DF, TEAM_DEF, PLAYER_POSITIONS, DEFENSE_VS_POS = load_data()
PREDICTORS = load_models()
PLAYERS = sorted(DF["PLAYER_NAME"].unique().tolist())

# Build player ID mapping from the data (column is "Player_ID" from NBA API)
PLAYER_IDS = {}
if "Player_ID" in DF.columns:
    for name in PLAYERS:
        player_rows = DF[DF["PLAYER_NAME"] == name]
        if len(player_rows) > 0:
            pid = player_rows["Player_ID"].iloc[0]
            if pid:
                PLAYER_IDS[name] = int(pid)

print(f"Loaded {len(DF)} game records for {len(PLAYERS)} players")
print(f"Player photos available: {len(PLAYER_IDS)}")

# =============================================================================
# COLOR SCHEME (Outlier Style)
# =============================================================================

COLORS = {
    "bg": "#0a0a0f",
    "card": "#12121a",
    "border": "#1e1e2e",
    "text": "#ffffff",
    "text_secondary": "#8888aa",
    "text_muted": "#555566",

    # Stat colors (from the screenshot)
    "pts": "#14b8a6",       # Teal for points
    "ast": "#f97066",       # Coral/orange for assists
    "reb": "#a78bfa",       # Purple for rebounds
    "blk": "#60a5fa",       # Blue for blocks
    "stl": "#fbbf24",       # Yellow/gold for steals
    "fg3m": "#ec4899",      # Pink for 3-pointers

    # Hit/miss colors for bars
    "hit": "#22c55e",       # Green for above threshold
    "miss": "#ef4444",      # Red for below threshold

    # Hit rate colors
    "hit_high": "#22c55e",   # Green 66%+
    "hit_mid": "#eab308",    # Yellow 50-65%
    "hit_low": "#ef4444",    # Red <50%

    # Accents
    "accent": "#14b8a6",
    "accent_secondary": "#8b5cf6",
}

def get_hit_color(pct):
    if pct >= 66:
        return COLORS["hit_high"]
    elif pct >= 50:
        return COLORS["hit_mid"]
    else:
        return COLORS["hit_low"]


def get_h2h_opponent(player_team):
    """Get today's opponent for a given team (for H2H mode)."""
    if not player_team:
        return ""

    teams_today = get_teams_playing_today()
    if player_team not in teams_today:
        return ""

    today_games = get_todays_games()
    if today_games.empty:
        return ""

    for _, game in today_games.iterrows():
        home = game.get("HOME_TEAM", "")
        away = game.get("AWAY_TEAM", "")
        if player_team == home:
            return away
        elif player_team == away:
            return home

    return ""


def filter_h2h_games(player_df, h2h_mode):
    """Filter player games by H2H opponent if in H2H mode. Returns (filtered_df, opponent_name)."""
    if h2h_mode != "h2h" or len(player_df) == 0:
        return player_df, ""

    player_team = player_df.iloc[0].get("TEAM_ABBREVIATION", "")
    opponent = get_h2h_opponent(player_team)

    if not opponent:
        return player_df.head(0), ""  # Return empty df

    def is_vs_opponent(matchup):
        if not isinstance(matchup, str):
            return False
        return opponent in matchup

    filtered_df = player_df[player_df["MATCHUP"].apply(is_vs_opponent)]
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

# Custom CSS to remove default browser styles and create seamless dark theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>NBA Props Dashboard</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            html, body {
                background-color: #0a0a0f;
                color: #ffffff;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                font-weight: 500;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                letter-spacing: -0.01em;
            }
            /* Typography defaults */
            h1, h2, h3, h4, h5, h6 {
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            p, span, div {
                font-weight: 500;
            }
            /* Remove Dash default margins */
            #react-entry-point, ._dash-loading {
                background-color: #0a0a0f !important;
            }
            /* Style dropdowns */
            .Select-control, .Select-menu-outer {
                background-color: #12121a !important;
                border-color: #1e1e2e !important;
                font-family: 'Inter', sans-serif !important;
                font-weight: 500 !important;
            }
            .Select-value-label, .Select-placeholder, .Select-input input {
                color: #ffffff !important;
                font-family: 'Inter', sans-serif !important;
                font-weight: 500 !important;
            }
            .Select-menu-outer {
                border-radius: 8px !important;
                margin-top: 4px !important;
            }
            .Select-option {
                background-color: #12121a !important;
                color: #ffffff !important;
                font-weight: 500 !important;
            }
            .Select-option:hover, .Select-option.is-focused {
                background-color: #1e1e2e !important;
            }
            .Select-arrow-zone {
                color: #8888aa !important;
            }
            /* Remove input outlines */
            input:focus, button:focus, .Select-control:focus {
                outline: none !important;
                box-shadow: none !important;
            }
            /* Button typography */
            button {
                font-family: 'Inter', sans-serif !important;
                font-weight: 600 !important;
            }
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #0a0a0f;
            }
            ::-webkit-scrollbar-thumb {
                background: #1e1e2e;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #2e2e3e;
            }
            /* Slider styling */
            .rc-slider-track {
                background-color: #14b8a6 !important;
            }
            .rc-slider-handle {
                border-color: #14b8a6 !important;
                background-color: #14b8a6 !important;
            }
            /* Responsive styles */
            @media (max-width: 1200px) {
                .main-container {
                    flex-direction: column !important;
                }
                .sidebar {
                    width: 100% !important;
                    margin-top: 20px;
                }
                .main-content {
                    margin-right: 0 !important;
                }
            }
            @media (max-width: 768px) {
                .nav-header {
                    flex-wrap: wrap;
                    gap: 10px;
                }
                .stat-tabs {
                    flex-wrap: wrap !important;
                    gap: 6px !important;
                }
                .period-tabs {
                    flex-wrap: wrap !important;
                    gap: 6px !important;
                }
                .supporting-stats {
                    flex-direction: column !important;
                }
                .player-header {
                    flex-direction: column !important;
                    text-align: center;
                }
                .player-photo {
                    margin: 0 auto 16px auto !important;
                }
                .Select {
                    width: 100% !important;
                }
            }
            @media (max-width: 480px) {
                body {
                    font-size: 14px;
                }
                .card {
                    padding: 16px !important;
                    border-radius: 12px !important;
                }
                button {
                    padding: 8px 12px !important;
                    font-size: 12px !important;
                }
            }
            .rc-slider-rail {
                background-color: #1e1e2e !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================================================
# STYLES
# =============================================================================

CARD = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "16px",
    "padding": "20px",
    "marginBottom": "16px",
    "border": "none",
    "boxShadow": "0 4px 20px rgba(0, 0, 0, 0.3)"
}

TAB_STYLE = {
    "padding": "10px 16px",
    "borderRadius": "8px",
    "cursor": "pointer",
    "fontSize": "13px",
    "fontWeight": "500",
    "color": COLORS["text_secondary"],
    "backgroundColor": "transparent",
    "border": "none",
    "marginRight": "6px",
    "whiteSpace": "nowrap",
    "transition": "all 0.2s ease",
}

TAB_ACTIVE = {
    **TAB_STYLE,
    "backgroundColor": "rgba(20, 184, 166, 0.15)",
    "color": COLORS["accent"],
    "fontWeight": "600",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_player_headshot_url(player_name):
    """Get NBA CDN headshot URL for a player"""
    player_id = PLAYER_IDS.get(player_name)
    if player_id:
        return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    return "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"

def get_stat_value(row, stat):
    """Calculate stat value for a row, handling combo stats"""
    if "+" in stat:
        parts = stat.split("+")
        return sum(row.get(p, 0) for p in parts)
    return row.get(stat, 0)

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

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div([
    # URL routing
    dcc.Location(id="url", refresh=False),

    # Navigation header
    html.Div([
        html.Div([
            # Logo/Title
            html.Div("NBA Props", style={
                "fontWeight": "700",
                "fontSize": "18px",
                "color": COLORS["text"],
                "marginRight": "32px"
            }),
            # Navigation links
            html.A("Player Analysis", href="/", style={
                "color": COLORS["text"],
                "textDecoration": "none",
                "padding": "8px 16px",
                "marginRight": "8px",
                "borderRadius": "6px",
                "fontSize": "14px"
            }, id="nav-player"),
            html.A("Today's Games", href="/games", style={
                "color": COLORS["text_muted"],
                "textDecoration": "none",
                "padding": "8px 16px",
                "borderRadius": "6px",
                "fontSize": "14px"
            }, id="nav-games"),
        ], style={"display": "flex", "alignItems": "center"})
    ], style={
        "padding": "12px 24px",
        "backgroundColor": COLORS["bg"],
        "borderBottom": f"1px solid {COLORS['border']}",
        "display": "flex",
        "alignItems": "center"
    }),

    # Page content container
    html.Div(id="page-content"),

], style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "color": COLORS["text"]})


def create_player_analysis_page():
    """Create the main player analysis page layout"""
    return html.Div([
    # Header with player photo and info
    html.Div([
        html.Div([
            # Player photo
            html.Div(id="player-photo", style={
                "width": "80px",
                "height": "80px",
                "borderRadius": "50%",
                "overflow": "hidden",
                "marginRight": "20px",
                "backgroundColor": COLORS["border"],
                "flexShrink": "0"
            }),

            # Player info and dropdown
            html.Div([
                dcc.Dropdown(
                    id="player-dropdown",
                    options=[{"label": p, "value": p} for p in PLAYERS],
                    value=PLAYERS[0] if PLAYERS else None,
                    placeholder="Search player...",
                    style={"width": "300px", "backgroundColor": COLORS["card"], "minWidth": "200px"}
                ),
                html.Div(id="player-header", style={"marginTop": "8px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "16px"}, className="player-header"),
    ], style={
        "padding": "20px 24px",
        "backgroundColor": COLORS["card"],
        "borderBottom": f"1px solid {COLORS['border']}"
    }),

    # Main content
    html.Div([
        # Left panel - Main chart area
        html.Div([
            # Stat type tabs (scrollable)
            html.Div([
                html.Div([
                    html.Button(
                        s["label"],
                        id=f"tab-{s['id'].lower().replace('+', '-')}",
                        n_clicks=0,
                        style=TAB_ACTIVE if s["id"] == "PTS" else TAB_STYLE
                    ) for s in STAT_TYPES
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}, className="stat-tabs")
            ], style={
                "overflowX": "auto",
                "marginBottom": "16px",
                "paddingBottom": "8px",
            }),

            # Store for selected stat
            dcc.Store(id="selected-stat", data="PTS"),

            # Time period tabs (with dynamic seasons)
            html.Div([
                html.Button("L5", id="period-l5", n_clicks=0, style=TAB_STYLE),
                html.Button("L10", id="period-l10", n_clicks=0, style=TAB_ACTIVE),
                html.Button("L20", id="period-l20", n_clicks=0, style=TAB_STYLE),
                html.Button("H2H", id="period-h2h", n_clicks=0, style=TAB_STYLE),
                html.Button(CURRENT_SEASON.split("-")[0], id="period-current", n_clicks=0, style=TAB_STYLE),
                html.Button(PREVIOUS_SEASON.split("-")[0], id="period-previous", n_clicks=0, style=TAB_STYLE),
            ], style={"display": "flex", "marginBottom": "20px", "flexWrap": "wrap", "gap": "6px"}, className="period-tabs"),

            dcc.Store(id="selected-period", data=10),
            dcc.Store(id="selected-season", data=None),
            dcc.Store(id="selected-h2h", data=None),  # Store for H2H opponent
            # Store the season values for use in callbacks
            dcc.Store(id="current-season-store", data=CURRENT_SEASON),
            dcc.Store(id="previous-season-store", data=PREVIOUS_SEASON),

            # Hit rate header with avg/median
            html.Div(id="hit-rate-header", style={"marginBottom": "20px"}),

            # Threshold slider
            html.Div([
                html.Div("Threshold:", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "12px",
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
                            0: {"label": "0", "style": {"color": "#fff"}},
                            10: {"label": "10", "style": {"color": "#fff"}},
                            20: {"label": "20", "style": {"color": "#fff"}},
                            30: {"label": "30", "style": {"color": "#fff"}},
                            40: {"label": "40", "style": {"color": "#fff"}},
                            50: {"label": "50", "style": {"color": "#fff"}},
                        },
                        included=True,
                        tooltip=None,
                    ),
                ], style={"flex": "1", "marginRight": "16px"}),
                html.Div(id="threshold-display", style={
                    "color": "#14b8a6",
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "minWidth": "60px",
                    "textAlign": "center",
                    "backgroundColor": "#000000",
                    "padding": "6px 12px",
                    "borderRadius": "6px",
                    "border": "2px solid #14b8a6"
                }),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

            # Main chart
            html.Div([
                dcc.Graph(id="main-chart", config={"displayModeBar": False})
            ], style=CARD),

            # Average/Median footer
            html.Div(id="avg-median-footer", style={
                "display": "flex",
                "justifyContent": "center",
                "gap": "40px",
                "padding": "12px",
                "backgroundColor": COLORS["card"],
                "borderRadius": "8px",
                "marginBottom": "16px"
            }),

            # Supporting Stats Section
            html.Div([
                # Header with Average/Median toggle
                html.Div([
                    html.Div("Supporting Stats", style={
                        "fontSize": "18px",
                        "fontWeight": "700",
                        "color": COLORS["text"]
                    }),
                    html.Div([
                        html.Button("Average", id="supporting-avg-btn", n_clicks=1, style={
                            "padding": "6px 16px",
                            "fontSize": "12px",
                            "backgroundColor": COLORS["border"],
                            "color": COLORS["text"],
                            "border": "none",
                            "borderRadius": "4px 0 0 4px",
                            "cursor": "pointer"
                        }),
                        html.Button("Median", id="supporting-median-btn", n_clicks=0, style={
                            "padding": "6px 16px",
                            "fontSize": "12px",
                            "backgroundColor": "transparent",
                            "color": COLORS["text_muted"],
                            "border": "none",
                            "borderRadius": "0 4px 4px 0",
                            "cursor": "pointer"
                        }),
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
                }, className="supporting-stats"),

                # Store for selected shooting stat
                dcc.Store(id="selected-shooting-stat", data="FG"),

                # Shooting breakdown chart (stacked bar)
                html.Div([
                    dcc.Graph(id="shooting-breakdown-chart", config={"displayModeBar": False})
                ]),

                # Insights section
                html.Div([
                    html.Div("Insights", style={
                        "color": COLORS["text"],
                        "fontSize": "18px",
                        "fontWeight": "700",
                        "marginBottom": "14px",
                        "marginTop": "24px",
                        "letterSpacing": "-0.02em"
                    }),
                    html.Div(id="player-insights", style={
                        "color": COLORS["text"],
                        "fontSize": "15px",
                        "fontWeight": "500",
                        "lineHeight": "1.7",
                        "padding": "18px",
                        "backgroundColor": COLORS["bg"],
                        "borderRadius": "12px",
                        "borderLeft": f"4px solid {COLORS['accent']}"
                    })
                ])
            ], style={**CARD, "marginTop": "16px"}),

            # Best Props for Today section
            html.Div([
                html.Div([
                    html.Span("🔥", style={"marginRight": "10px", "fontSize": "20px"}),
                    html.Span("BEST PROPS FOR TODAY", style={
                        "color": COLORS["text"],
                        "fontSize": "18px",
                        "fontWeight": "700",
                        "letterSpacing": "-0.02em"
                    })
                ], style={"marginBottom": "16px", "display": "flex", "alignItems": "center"}),
                html.Div(id="best-props-main", style={
                    "maxHeight": "400px",
                    "overflowY": "auto"
                })
            ], style={**CARD, "marginTop": "16px"}),

        ], style={"flex": "2", "marginRight": "20px"}, className="main-content"),

        # Right panel - Sidebar
        html.Div([
            # Tabs for sidebar sections
            html.Div([
                html.Button("Matchup", id="sidebar-matchup", n_clicks=1, style=TAB_ACTIVE),
                html.Button("Injuries", id="sidebar-injuries", n_clicks=0, style=TAB_STYLE),
                html.Button("Insights", id="sidebar-insights", n_clicks=0, style=TAB_STYLE),
                html.Button("Best Props", id="sidebar-props", n_clicks=0, style=TAB_STYLE),
            ], style={"display": "flex", "marginBottom": "16px", "flexWrap": "wrap", "gap": "6px"}, className="period-tabs"),

            dcc.Store(id="sidebar-tab", data="matchup"),

            # Dynamic sidebar content
            html.Div(id="sidebar-content"),

        ], style={"width": "380px", "flexShrink": "0"}, className="sidebar"),

    ], style={
        "display": "flex",
        "padding": "24px",
        "maxWidth": "1600px",
        "margin": "0 auto",
        "flexWrap": "wrap"
    }, className="main-container"),

    # Auto-refresh interval (every 5 minutes)
    dcc.Interval(
        id="auto-refresh-interval",
        interval=5 * 60 * 1000,  # 5 minutes in milliseconds
        n_intervals=0
    ),

    # Store for last update time
    dcc.Store(id="last-update-time", data=datetime.now().isoformat()),

    ])


def create_todays_games_page():
    """Create the Today's Games page showing all matchups"""
    from utils.data_fetch import get_todays_games

    # Team abbreviation to ID mapping for logos
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

    games = get_todays_games()

    if games.empty:
        return html.Div([
            html.Div([
                html.Div("TODAY'S GAMES", style={
                    "fontSize": "24px",
                    "fontWeight": "700",
                    "marginBottom": "8px"
                }),
                html.Div(datetime.now().strftime("%A, %B %d, %Y"), style={
                    "color": COLORS["text_muted"],
                    "fontSize": "14px",
                    "marginBottom": "32px"
                }),
                html.Div("No games scheduled today", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "16px",
                    "textAlign": "center",
                    "padding": "60px 0"
                })
            ], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "32px 24px"})
        ])

    # Build game cards
    game_cards = []
    for _, game in games.iterrows():
        # Get team abbreviations - handle different column names
        home_team = game.get("HOME_TEAM", game.get("HOME_TEAM_ABBREVIATION", ""))
        away_team = game.get("AWAY_TEAM", game.get("VISITOR_TEAM_ABBREVIATION", game.get("VISITOR_TEAM", "")))
        game_status = game.get("GAME_STATUS_TEXT", "Scheduled")

        # Get team logo URLs
        home_logo = get_team_logo_url(home_team)
        away_logo = get_team_logo_url(away_team)

        # Debug: print what we found
        # print(f"Game: {away_team} @ {home_team}")

        # Get team defensive stats
        home_def = TEAM_DEF[TEAM_DEF["TEAM_ABBREVIATION"] == home_team] if "TEAM_ABBREVIATION" in TEAM_DEF.columns else pd.DataFrame()
        away_def = TEAM_DEF[TEAM_DEF["TEAM_ABBREVIATION"] == away_team] if "TEAM_ABBREVIATION" in TEAM_DEF.columns else pd.DataFrame()

        # Get defensive rankings
        home_pts_allowed = home_def.iloc[0].get("OPP_PTS", 0) if len(home_def) > 0 else 0
        away_pts_allowed = away_def.iloc[0].get("OPP_PTS", 0) if len(away_def) > 0 else 0

        # Get position-specific defense
        home_guard_def = DEFENSE_VS_POS[(DEFENSE_VS_POS["TEAM_ABBREVIATION"] == home_team) & (DEFENSE_VS_POS["POSITION"] == "G")] if not DEFENSE_VS_POS.empty else pd.DataFrame()
        home_forward_def = DEFENSE_VS_POS[(DEFENSE_VS_POS["TEAM_ABBREVIATION"] == home_team) & (DEFENSE_VS_POS["POSITION"] == "F")] if not DEFENSE_VS_POS.empty else pd.DataFrame()
        home_center_def = DEFENSE_VS_POS[(DEFENSE_VS_POS["TEAM_ABBREVIATION"] == home_team) & (DEFENSE_VS_POS["POSITION"] == "C")] if not DEFENSE_VS_POS.empty else pd.DataFrame()

        away_guard_def = DEFENSE_VS_POS[(DEFENSE_VS_POS["TEAM_ABBREVIATION"] == away_team) & (DEFENSE_VS_POS["POSITION"] == "G")] if not DEFENSE_VS_POS.empty else pd.DataFrame()
        away_forward_def = DEFENSE_VS_POS[(DEFENSE_VS_POS["TEAM_ABBREVIATION"] == away_team) & (DEFENSE_VS_POS["POSITION"] == "F")] if not DEFENSE_VS_POS.empty else pd.DataFrame()
        away_center_def = DEFENSE_VS_POS[(DEFENSE_VS_POS["TEAM_ABBREVIATION"] == away_team) & (DEFENSE_VS_POS["POSITION"] == "C")] if not DEFENSE_VS_POS.empty else pd.DataFrame()

        def get_rank_badge(rank, label):
            if pd.isna(rank):
                return html.Span()
            rank = int(rank)
            color = COLORS["hit_high"] if rank <= 10 else COLORS["hit_mid"] if rank <= 20 else COLORS["text_muted"]
            return html.Div([
                html.Span(f"#{rank}", style={"fontWeight": "600", "color": color}),
                html.Span(f" {label}", style={"fontSize": "10px", "color": COLORS["text_muted"]})
            ], style={"marginRight": "12px", "fontSize": "12px"})

        game_card = html.Div([
            # Game header with logos
            html.Div([
                # Away team with logo
                html.Div([
                    html.Img(src=away_logo, style={
                        "width": "48px",
                        "height": "48px",
                        "marginRight": "12px"
                    }) if away_logo else html.Div(style={"width": "48px", "height": "48px", "marginRight": "12px"}),
                    html.Span(away_team, style={"fontWeight": "700", "fontSize": "22px"})
                ], style={"display": "flex", "alignItems": "center"}),

                # @ symbol
                html.Span("@", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "18px",
                    "margin": "0 20px"
                }),

                # Home team with logo
                html.Div([
                    html.Span(home_team, style={"fontWeight": "700", "fontSize": "22px"}),
                    html.Img(src=home_logo, style={
                        "width": "48px",
                        "height": "48px",
                        "marginLeft": "12px"
                    }) if home_logo else html.Div(style={"width": "48px", "height": "48px", "marginLeft": "12px"})
                ], style={"display": "flex", "alignItems": "center"}),

                # Game status
                html.Div(game_status, style={
                    "color": COLORS["accent"],
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "marginLeft": "auto",
                    "padding": "6px 12px",
                    "backgroundColor": COLORS["bg"],
                    "borderRadius": "4px"
                })
            ], style={
                "display": "flex",
                "alignItems": "center",
                "marginBottom": "24px"
            }),

            # Two columns: Away team vs Home team
            html.Div([
                # Away team column
                html.Div([
                    html.Div([
                        html.Img(src=away_logo, style={
                            "width": "24px",
                            "height": "24px",
                            "marginRight": "8px"
                        }) if away_logo else None,
                        html.Span(f"{away_team} Defense", style={
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": COLORS["text"]
                        })
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
                    html.Div([
                        html.Div([
                            html.Div("vs Guards", style={"fontSize": "11px", "color": COLORS["text_muted"], "marginBottom": "4px"}),
                            html.Div([
                                get_rank_badge(away_guard_def.iloc[0].get("PTS_RANK") if len(away_guard_def) > 0 else None, "PTS"),
                                get_rank_badge(away_guard_def.iloc[0].get("AST_RANK") if len(away_guard_def) > 0 else None, "AST"),
                            ], style={"display": "flex"})
                        ], style={"marginBottom": "10px"}),
                        html.Div([
                            html.Div("vs Forwards", style={"fontSize": "11px", "color": COLORS["text_muted"], "marginBottom": "4px"}),
                            html.Div([
                                get_rank_badge(away_forward_def.iloc[0].get("PTS_RANK") if len(away_forward_def) > 0 else None, "PTS"),
                                get_rank_badge(away_forward_def.iloc[0].get("REB_RANK") if len(away_forward_def) > 0 else None, "REB"),
                            ], style={"display": "flex"})
                        ], style={"marginBottom": "10px"}),
                        html.Div([
                            html.Div("vs Centers", style={"fontSize": "11px", "color": COLORS["text_muted"], "marginBottom": "4px"}),
                            html.Div([
                                get_rank_badge(away_center_def.iloc[0].get("PTS_RANK") if len(away_center_def) > 0 else None, "PTS"),
                                get_rank_badge(away_center_def.iloc[0].get("REB_RANK") if len(away_center_def) > 0 else None, "REB"),
                            ], style={"display": "flex"})
                        ]),
                    ]),
                    html.Div(f"PPG Allowed: {away_pts_allowed:.1f}" if away_pts_allowed else "", style={
                        "fontSize": "12px",
                        "color": COLORS["text_muted"],
                        "marginTop": "12px"
                    })
                ], style={"flex": "1", "paddingRight": "20px"}),

                # Divider
                html.Div(style={
                    "width": "1px",
                    "backgroundColor": COLORS["border"],
                    "margin": "0 20px"
                }),

                # Home team column
                html.Div([
                    html.Div([
                        html.Img(src=home_logo, style={
                            "width": "24px",
                            "height": "24px",
                            "marginRight": "8px"
                        }) if home_logo else None,
                        html.Span(f"{home_team} Defense", style={
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": COLORS["text"]
                        })
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
                    html.Div([
                        html.Div([
                            html.Div("vs Guards", style={"fontSize": "11px", "color": COLORS["text_muted"], "marginBottom": "4px"}),
                            html.Div([
                                get_rank_badge(home_guard_def.iloc[0].get("PTS_RANK") if len(home_guard_def) > 0 else None, "PTS"),
                                get_rank_badge(home_guard_def.iloc[0].get("AST_RANK") if len(home_guard_def) > 0 else None, "AST"),
                            ], style={"display": "flex"})
                        ], style={"marginBottom": "10px"}),
                        html.Div([
                            html.Div("vs Forwards", style={"fontSize": "11px", "color": COLORS["text_muted"], "marginBottom": "4px"}),
                            html.Div([
                                get_rank_badge(home_forward_def.iloc[0].get("PTS_RANK") if len(home_forward_def) > 0 else None, "PTS"),
                                get_rank_badge(home_forward_def.iloc[0].get("REB_RANK") if len(home_forward_def) > 0 else None, "REB"),
                            ], style={"display": "flex"})
                        ], style={"marginBottom": "10px"}),
                        html.Div([
                            html.Div("vs Centers", style={"fontSize": "11px", "color": COLORS["text_muted"], "marginBottom": "4px"}),
                            html.Div([
                                get_rank_badge(home_center_def.iloc[0].get("PTS_RANK") if len(home_center_def) > 0 else None, "PTS"),
                                get_rank_badge(home_center_def.iloc[0].get("REB_RANK") if len(home_center_def) > 0 else None, "REB"),
                            ], style={"display": "flex"})
                        ]),
                    ]),
                    html.Div(f"PPG Allowed: {home_pts_allowed:.1f}" if home_pts_allowed else "", style={
                        "fontSize": "12px",
                        "color": COLORS["text_muted"],
                        "marginTop": "12px"
                    })
                ], style={"flex": "1", "paddingLeft": "20px"}),
            ], style={"display": "flex"}),

        ], style={
            "padding": "24px",
            "backgroundColor": COLORS["card"],
            "borderRadius": "12px",
            "marginBottom": "16px",
            "border": f"1px solid {COLORS['border']}"
        })

        game_cards.append(game_card)

    return html.Div([
        html.Div([
            html.Div("TODAY'S GAMES", style={
                "fontSize": "24px",
                "fontWeight": "700",
                "marginBottom": "8px"
            }),
            html.Div([
                html.Span(datetime.now().strftime("%A, %B %d, %Y"), style={
                    "color": COLORS["text_muted"],
                    "fontSize": "14px",
                }),
                html.Span(f" • {len(games)} games", style={
                    "color": COLORS["accent"],
                    "fontSize": "14px",
                    "marginLeft": "8px"
                })
            ], style={"marginBottom": "32px"}),

            # Legend
            html.Div([
                html.Span("Rankings: ", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
                html.Span("#1-10 ", style={"color": COLORS["hit_high"], "fontSize": "11px", "fontWeight": "600"}),
                html.Span("= favorable matchup | ", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
                html.Span("#11-20 ", style={"color": COLORS["hit_mid"], "fontSize": "11px", "fontWeight": "600"}),
                html.Span("= average | ", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
                html.Span("#21-30 ", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Span("= tough matchup", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
            ], style={"marginBottom": "24px"}),

            # Game cards
            html.Div(game_cards),

            # Last updated
            html.Div(f"Last updated: {datetime.now().strftime('%I:%M %p')}", style={
                "color": COLORS["text_muted"],
                "fontSize": "11px",
                "textAlign": "center",
                "marginTop": "24px"
            })
        ], style={"maxWidth": "1000px", "margin": "0 auto", "padding": "32px 24px"})
    ])


# =============================================================================
# PAGE ROUTING
# =============================================================================

@callback(
    [Output("page-content", "children"),
     Output("nav-player", "style"),
     Output("nav-games", "style")],
    Input("url", "pathname")
)
def display_page(pathname):
    """Route to the correct page based on URL"""
    nav_active = {
        "color": COLORS["text"],
        "textDecoration": "none",
        "padding": "8px 16px",
        "marginRight": "8px",
        "borderRadius": "6px",
        "fontSize": "14px",
        "backgroundColor": COLORS["border"]
    }
    nav_inactive = {
        "color": COLORS["text_muted"],
        "textDecoration": "none",
        "padding": "8px 16px",
        "marginRight": "8px",
        "borderRadius": "6px",
        "fontSize": "14px"
    }

    if pathname == "/games":
        return create_todays_games_page(), nav_inactive, nav_active
    else:
        return create_player_analysis_page(), nav_active, nav_inactive


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
    [Output(f"tab-{s['id'].lower().replace('+', '-')}", "style") for s in STAT_TYPES],
    Input("selected-stat", "data")
)
def update_stat_tab_styles(selected):
    styles = []
    for s in STAT_TYPES:
        if s["id"] == selected:
            styles.append(TAB_ACTIVE)
        else:
            styles.append(TAB_STYLE)
    return styles


# Tab click handlers for period
@callback(
    [Output("selected-period", "data"),
     Output("selected-season", "data"),
     Output("selected-h2h", "data"),
     Output("period-l5", "style"),
     Output("period-l10", "style"),
     Output("period-l20", "style"),
     Output("period-h2h", "style"),
     Output("period-current", "style"),
     Output("period-previous", "style")],
    [Input("period-l5", "n_clicks"),
     Input("period-l10", "n_clicks"),
     Input("period-l20", "n_clicks"),
     Input("period-h2h", "n_clicks"),
     Input("period-current", "n_clicks"),
     Input("period-previous", "n_clicks")]
)
def update_period_tabs(l5, l10, l20, h2h, current, previous):
    from dash import ctx
    triggered = ctx.triggered_id

    styles = [TAB_STYLE] * 6
    period = 10
    season = None
    h2h_mode = None  # Will be set to "h2h" when H2H is selected

    if triggered == "period-l5":
        period, styles[0] = 5, TAB_ACTIVE
    elif triggered == "period-l10" or triggered is None:
        period, styles[1] = 10, TAB_ACTIVE
    elif triggered == "period-l20":
        period, styles[2] = 20, TAB_ACTIVE
    elif triggered == "period-h2h":
        period, h2h_mode, styles[3] = 100, "h2h", TAB_ACTIVE  # Use large period, filter by opponent
    elif triggered == "period-current":
        period, season, styles[4] = 100, CURRENT_SEASON, TAB_ACTIVE
    elif triggered == "period-previous":
        period, season, styles[5] = 100, PREVIOUS_SEASON, TAB_ACTIVE

    return [period, season, h2h_mode] + styles


# Sidebar tab handler
@callback(
    [Output("sidebar-tab", "data"),
     Output("sidebar-matchup", "style"),
     Output("sidebar-injuries", "style"),
     Output("sidebar-insights", "style"),
     Output("sidebar-props", "style")],
    [Input("sidebar-matchup", "n_clicks"),
     Input("sidebar-injuries", "n_clicks"),
     Input("sidebar-insights", "n_clicks"),
     Input("sidebar-props", "n_clicks")]
)
def update_sidebar_tabs(matchup, injuries, insights, props):
    from dash import ctx
    triggered = ctx.triggered_id

    styles = [TAB_STYLE] * 4
    tab = "matchup"

    if triggered == "sidebar-matchup" or triggered is None:
        tab, styles[0] = "matchup", TAB_ACTIVE
    elif triggered == "sidebar-injuries":
        tab, styles[1] = "injuries", TAB_ACTIVE
    elif triggered == "sidebar-insights":
        tab, styles[2] = "insights", TAB_ACTIVE
    elif triggered == "sidebar-props":
        tab, styles[3] = "props", TAB_ACTIVE

    return [tab] + styles


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
     Input("threshold-slider", "value")]
)
def update_hit_rate_header(player_name, stat, period, h2h_mode, threshold):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # H2H mode: filter by today's opponent
    h2h_opponent = ""
    if h2h_mode == "h2h":
        player_df, h2h_opponent = filter_h2h_games(player_df, h2h_mode)

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
    hit_pct, hits = calc_hit(player_df, period)
    l5_pct, _ = calc_hit(player_df, 5)
    l20_pct, _ = calc_hit(player_df, 20)

    # Season hit rates (dynamic)
    df_current = player_df[player_df["SEASON"] == CURRENT_SEASON]
    df_previous = player_df[player_df["SEASON"] == PREVIOUS_SEASON]
    pct_current, _ = calc_hit(df_current, len(df_current)) if len(df_current) > 0 else (0, 0)
    pct_previous, _ = calc_hit(df_previous, len(df_previous)) if len(df_previous) > 0 else (0, 0)

    return html.Div([
        html.Div([
            html.Span("% ", style={"color": COLORS["accent"], "fontSize": "18px"}),
            html.Span(f"{player_name} - {stat_display}", style={"fontSize": "16px", "fontWeight": "500"}),
        ], style={"marginBottom": "12px"}),

        html.Div([
            # Main hit rate
            html.Div([
                html.Div(f"Last {min(period, len(player_df))}", style={"color": COLORS["text_secondary"], "fontSize": "12px"}),
                html.Div([
                    html.Span(f"{hit_pct:.0f}%", style={
                        "color": get_hit_color(hit_pct),
                        "fontSize": "24px",
                        "fontWeight": "600"
                    }),
                    html.Span(f" {hits} of {min(period, len(player_df))}", style={
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
     Input("selected-h2h", "data")]
)
def update_avg_median(player_name, stat, period, h2h_mode):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # H2H mode: filter by today's opponent
    h2h_opponent = ""
    if h2h_mode == "h2h":
        player_df, h2h_opponent = filter_h2h_games(player_df, h2h_mode)
    else:
        player_df = player_df.head(period)

    if "+" in stat:
        parts = stat.split("+")
        vals = sum(player_df[p] for p in parts if p in player_df.columns)
    else:
        vals = player_df[stat] if stat in player_df.columns else pd.Series([0])

    avg = vals.mean() if len(vals) > 0 else 0
    median = vals.median() if len(vals) > 0 else 0

    label = f"vs {h2h_opponent}" if h2h_mode == "h2h" and h2h_opponent else f"L{period}"

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
     Input("threshold-slider", "value")]
)
def update_main_chart(player_name, stat, period, season, h2h_mode, threshold):
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # H2H mode: filter by today's opponent
    if h2h_mode == "h2h":
        player_df, _ = filter_h2h_games(player_df, h2h_mode)
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
        line_dash="solid",
        line_color=COLORS["text_secondary"],
        line_width=2,
        annotation_text=f"{threshold}",
        annotation_position="left",
        annotation_font_color=COLORS["text_secondary"]
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
     Input("selected-stat", "data"),
     Input("auto-refresh-interval", "n_intervals")]
)
def update_sidebar_content(tab, player_name, stat, n_intervals):
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
     Output("supporting-avg-btn", "style"),
     Output("supporting-median-btn", "style")],
    [Input("supporting-avg-btn", "n_clicks"),
     Input("supporting-median-btn", "n_clicks")]
)
def update_supporting_stat_mode(_avg_clicks, _median_clicks):
    from dash import ctx
    triggered = ctx.triggered_id

    active_style = {
        "padding": "6px 16px",
        "fontSize": "12px",
        "backgroundColor": COLORS["border"],
        "color": COLORS["text"],
        "border": "none",
        "borderRadius": "4px 0 0 4px",
        "cursor": "pointer"
    }
    inactive_style = {
        "padding": "6px 16px",
        "fontSize": "12px",
        "backgroundColor": "transparent",
        "color": COLORS["text_muted"],
        "border": "none",
        "borderRadius": "0 4px 4px 0",
        "cursor": "pointer"
    }

    if triggered == "supporting-median-btn":
        return "median", {**inactive_style, "borderRadius": "4px 0 0 4px"}, {**active_style, "borderRadius": "0 4px 4px 0"}
    else:
        return "average", active_style, inactive_style


@callback(
    [Output("supporting-stats-cards", "children"),
     Output("selected-shooting-stat", "data")],
    [Input("player-dropdown", "value"),
     Input("supporting-stat-mode", "data"),
     Input("selected-period", "data"),
     Input("selected-season", "data"),
     Input("selected-h2h", "data"),
     Input("selected-shooting-stat", "data")]
)
def update_supporting_stats_cards(player_name, mode, period, season, h2h_mode, current_selected):
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

    # H2H mode: filter by today's opponent
    if h2h_mode == "h2h":
        player_df, _ = filter_h2h_games(player_df, h2h_mode)
    elif season:
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

    # Build stat cards - all clickable now
    cards = []

    for stat in stats:
        is_selected = stat["key"] == selected_stat

        card_style = {
            "padding": "12px 16px",
            "backgroundColor": COLORS["card"] if is_selected else COLORS["bg"],
            "borderRadius": "8px",
            "minWidth": "100px",
            "cursor": "pointer",
            "border": f"2px solid {COLORS['text']}" if is_selected else f"1px solid {COLORS['border']}",
            "transition": "all 0.2s ease"
        }

        if stat["type"] == "shooting":
            # Shooting stat with percentage
            card_content = html.Button([
                html.Div(stat["label"], style={
                    "color": COLORS["text"] if is_selected else COLORS["text_muted"],
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "marginBottom": "4px"
                }),
                html.Div(f"{stat['value']:.1f} Made ({stat['pct']:.0f}%)", style={
                    "color": COLORS["text"],
                    "fontSize": "13px",
                    "fontWeight": "500"
                }),
                html.Div(f"{stat['attempts']:.1f} Attempts", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "11px"
                }),
            ], id={"type": "supporting-stat-btn", "index": stat["key"]}, n_clicks=0, style={**card_style, "textAlign": "left", "width": "auto"})
        else:
            # Simple stat (Minutes, Fouls, AST)
            card_content = html.Button([
                html.Div(stat["label"], style={
                    "color": COLORS["text"] if is_selected else COLORS["text_muted"],
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "marginBottom": "4px"
                }),
                html.Div(f"{stat['value']:.1f}", style={
                    "color": COLORS["text"],
                    "fontSize": "18px",
                    "fontWeight": "600"
                }),
            ], id={"type": "supporting-stat-btn", "index": stat["key"]}, n_clicks=0, style={**card_style, "textAlign": "left", "width": "auto"})

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
     Input("selected-season", "data"),
     Input("selected-h2h", "data")]
)
def update_shooting_breakdown_chart(player_name, selected_stat, period, season, h2h_mode):
    """Create bar chart showing stats per game - stacked for shooting, simple for other stats"""
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

    # H2H mode: filter by today's opponent
    if h2h_mode == "h2h":
        player_df, _ = filter_h2h_games(player_df, h2h_mode)
    elif season:
        player_df = player_df[player_df["SEASON"] == season]
    elif period:
        player_df = player_df.head(period)

    player_df = player_df.iloc[::-1]  # Reverse for chronological order

    if len(player_df) == 0:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            height=300
        )
        return fig

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
        # Stacked bar chart for shooting stats
        made_col, attempts_col, title = shooting_stats[selected_stat]

        if made_col not in player_df.columns or attempts_col not in player_df.columns:
            return fig

        made = player_df[made_col].fillna(0).astype(int)
        attempts = player_df[attempts_col].fillna(0).astype(int)
        missed = attempts - made
        pct = (made / attempts * 100).replace([np.inf, -np.inf], 0).fillna(0)

        # Made bars (bottom)
        fig.add_trace(go.Bar(
            x=list(range(len(player_df))),
            y=made,
            name="Made",
            marker_color="#4a5568",
            text=[f"{m}<br>Made" for m in made],
            textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate="Made: %{y}<extra></extra>"
        ))

        # Missed bars (top, stacked)
        fig.add_trace(go.Bar(
            x=list(range(len(player_df))),
            y=missed,
            name="Missed",
            marker_color="#718096",
            text=[f"{int(p)}%" for p in pct],
            textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate="Missed: %{y}<extra></extra>"
        ))

        # Add total labels on top
        fig.add_trace(go.Scatter(
            x=list(range(len(player_df))),
            y=attempts + 0.5,
            mode="text",
            text=[str(a) for a in attempts],
            textposition="top center",
            textfont=dict(size=12, color=COLORS["text"]),
            hoverinfo="skip",
            showlegend=False
        ))

        fig.update_layout(barmode="stack")

    elif selected_stat in simple_stats:
        # Simple bar chart for non-shooting stats
        col, title = simple_stats[selected_stat]

        if col not in player_df.columns:
            return fig

        values = player_df[col].fillna(0)
        avg_val = values.mean()

        # Color bars based on above/below average
        bar_colors = [COLORS["hit_high"] if v >= avg_val else COLORS["hit_low"] for v in values]

        fig.add_trace(go.Bar(
            x=list(range(len(player_df))),
            y=values,
            marker_color=bar_colors,
            text=[f"{v:.0f}" if col != "MIN" else f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate=f"{title}: %{{y}}<extra></extra>"
        ))

        # Add average line
        fig.add_hline(
            y=avg_val,
            line_dash="dash",
            line_color=COLORS["text_secondary"],
            line_width=2,
            annotation_text=f"Avg: {avg_val:.1f}",
            annotation_position="left",
            annotation_font_color=COLORS["text_secondary"]
        )
    else:
        # Fallback to FG
        return update_shooting_breakdown_chart(player_name, "FG", period, season)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=40, r=20, t=40, b=60),
        height=320,
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(player_df))),
            ticktext=labels,
            tickfont=dict(size=10, color=COLORS["text_muted"]),
            showgrid=False
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            tickfont=dict(size=10, color=COLORS["text_muted"]),
            showgrid=True
        ),
        bargap=0.3
    )

    return fig


@callback(
    Output("player-insights", "children"),
    [Input("player-dropdown", "value"),
     Input("selected-period", "data"),
     Input("selected-season", "data"),
     Input("selected-h2h", "data")]
)
def generate_player_insights(player_name, period, season, h2h_mode):
    """Generate smart insights about player performance trends"""
    if not player_name:
        return "Select a player to see insights."

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # H2H mode: filter by today's opponent
    h2h_opponent = ""
    if h2h_mode == "h2h":
        player_df, h2h_opponent = filter_h2h_games(player_df, h2h_mode)
        if len(player_df) < 2:
            return f"Not enough H2H games vs {h2h_opponent if h2h_opponent else 'opponent'} to generate insights."
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


@callback(
    Output("best-props-main", "children"),
    [Input("player-dropdown", "value")]
)
def update_best_props_main(selected_player):
    """Generate best props for today's games - shown in main content area"""
    today_games = get_todays_games()

    if today_games.empty:
        return html.Div("No games scheduled today", style={
            "color": COLORS["text_muted"],
            "textAlign": "center",
            "padding": "30px"
        })

    # Build a mapping of team -> opponent for today's games
    team_to_opponent = {}
    for _, game in today_games.iterrows():
        home = game.get("HOME_TEAM", "")
        away = game.get("AWAY_TEAM", "")
        if home and away:
            team_to_opponent[home] = away
            team_to_opponent[away] = home

    teams_today = set(team_to_opponent.keys())

    if not teams_today:
        return html.Div("No games scheduled today", style={
            "color": COLORS["text_muted"],
            "textAlign": "center",
            "padding": "30px"
        })

    # Get players on teams playing today (exclude injured)
    best_props = []

    for player_name in PLAYERS[:100]:  # Check top 100 players
        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)
        if len(player_df) < 5:
            continue

        # Get player's CURRENT team from PLAYER_POSITIONS (most accurate source)
        player_team = ""
        player_position = "G"
        if not PLAYER_POSITIONS.empty:
            pos_match = PLAYER_POSITIONS[PLAYER_POSITIONS["PLAYER_NAME"] == player_name]
            if len(pos_match) > 0:
                player_team = str(pos_match["TEAM_ABBREVIATION"].iloc[0])
                pos = str(pos_match["POSITION"].iloc[0])
                if "G" in pos:
                    player_position = "G"
                elif "F" in pos:
                    player_position = "F"
                elif "C" in pos:
                    player_position = "C"

        # Skip if we couldn't find the player's current team
        if not player_team:
            continue

        # Skip if player's team isn't playing today
        if player_team not in teams_today:
            continue

        # Get TODAY's actual opponent from today's schedule
        opponent = team_to_opponent.get(player_team, "")
        if not opponent:
            continue

        # Check injury status
        try:
            injury_status = get_player_injury_status(player_name)
            if injury_status.get("status") == "OUT":
                continue
        except:
            pass

        # Get defensive ranking for TODAY's opponent
        opp_def_rank = 15
        if not DEFENSE_VS_POS.empty and opponent:
            opp_def = DEFENSE_VS_POS[
                (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opponent) &
                (DEFENSE_VS_POS["POSITION"] == player_position)
            ]
            if len(opp_def) > 0:
                opp_def_rank = int(opp_def["PTS_RANK"].iloc[0])

        # Calculate stats from last 10 games
        l10 = player_df.head(10)
        pts_avg = l10["PTS"].mean()

        # Calculate hit rates
        pts_line = round(pts_avg * 0.9, 1)
        pts_hits = (l10["PTS"] > pts_line).sum()
        pts_hit_pct = int(pts_hits / len(l10) * 100)

        # Score based on hit rate and matchup
        score = pts_hit_pct
        if opp_def_rank >= 20:  # Weak defense
            score += 15
        elif opp_def_rank <= 6:  # Elite defense
            score -= 10

        # Determine confidence
        if score >= 85:
            confidence = "HIGH"
            conf_color = COLORS["hit_high"]
        elif score >= 70:
            confidence = "MED"
            conf_color = COLORS["hit_mid"]
        else:
            confidence = "LOW"
            conf_color = COLORS["hit_low"]

        # Only include if decent hit rate
        if pts_hit_pct >= 60:
            pos_name = {"G": "guards", "F": "forwards", "C": "centers"}.get(player_position, "players")
            reason = f"vs {opponent} (#{opp_def_rank} vs {pos_name}) • {pts_hit_pct}% hit rate L10"

            best_props.append({
                "player": player_name,
                "team": player_team,
                "prop": f"Over {pts_line} PTS",
                "projection": pts_avg,
                "hit_rate": pts_hit_pct,
                "confidence": confidence,
                "conf_color": conf_color,
                "reason": reason,
                "score": score,
                "opponent": opponent,
                "def_rank": opp_def_rank
            })

    # Sort by score
    best_props.sort(key=lambda x: x["score"], reverse=True)

    if not best_props:
        return html.Div("No strong props found for today", style={
            "color": COLORS["text_muted"],
            "textAlign": "center",
            "padding": "30px"
        })

    # Build UI for top 5 props
    prop_cards = []
    for prop in best_props[:5]:
        prop_cards.append(
            html.Div([
                # Header row
                html.Div([
                    html.Div([
                        html.Span(prop["player"], style={
                            "fontWeight": "600",
                            "fontSize": "14px",
                            "color": COLORS["text"]
                        }),
                        html.Span(f" vs {prop['opponent']}", style={
                            "color": COLORS["text_muted"],
                            "fontSize": "13px"
                        })
                    ]),
                    html.Span(prop["confidence"], style={
                        "backgroundColor": prop["conf_color"],
                        "color": "#000" if prop["confidence"] == "MED" else "#fff",
                        "padding": "2px 8px",
                        "borderRadius": "4px",
                        "fontSize": "11px",
                        "fontWeight": "700"
                    })
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"}),

                # Prop line
                html.Div(prop["prop"], style={
                    "color": COLORS["accent"],
                    "fontSize": "16px",
                    "fontWeight": "700",
                    "marginBottom": "6px"
                }),

                # Reason
                html.Div(prop["reason"], style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "12px"
                })
            ], style={
                "padding": "14px",
                "backgroundColor": COLORS["bg"],
                "borderRadius": "10px",
                "marginBottom": "10px",
                "borderLeft": f"3px solid {prop['conf_color']}"
            })
        )

    return html.Div(prop_cards)


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

    # Get position-specific defensive stats from DEFENSE_VS_POS
    pts_allowed = 0
    ast_allowed = 0
    reb_allowed = 0
    pts_rank = 15
    ast_rank = 15
    reb_rank = 15

    if not DEFENSE_VS_POS.empty and opponent:
        # Filter for opponent team and player's position
        pos_def = DEFENSE_VS_POS[
            (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opponent) &
            (DEFENSE_VS_POS["POSITION"] == player_position)
        ]
        if len(pos_def) > 0:
            row = pos_def.iloc[0]
            pts_allowed = row.get("AVG_PTS_ALLOWED", 0)
            ast_allowed = row.get("AVG_AST_ALLOWED", 0)
            reb_allowed = row.get("AVG_REB_ALLOWED", 0)
            pts_rank = int(row.get("PTS_RANK", 15))
            ast_rank = int(row.get("AST_RANK", 15))
            reb_rank = int(row.get("REB_RANK", 15))

    # Fallback to team-level defense stats if position-specific not available
    if pts_allowed == 0 and not TEAM_DEF.empty:
        opp_def = TEAM_DEF[TEAM_DEF["TEAM_ABBREVIATION"] == opponent] if "TEAM_ABBREVIATION" in TEAM_DEF.columns else pd.DataFrame()
        if not opp_def.empty:
            row = opp_def.iloc[0]
            pts_allowed = row.get("OPP_PTS", row.get("PTS", 0))
            ast_allowed = row.get("OPP_AST", row.get("AST", 0))
            reb_allowed = row.get("OPP_REB", row.get("REB", 0))
            if "OPP_PTS" in TEAM_DEF.columns:
                pts_rank = int(TEAM_DEF["OPP_PTS"].rank(ascending=False).loc[opp_def.index[0]])
                ast_rank = int(TEAM_DEF["OPP_AST"].rank(ascending=False).loc[opp_def.index[0]])

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
            html.Div(f"Key {opponent} Defense vs {player_pos_display}", style={
                "color": COLORS["text"],
                "fontSize": "14px",
                "fontWeight": "600",
                "marginBottom": "16px"
            }),

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

            # Stats table
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Stat (per game)", style={"textAlign": "left", "color": COLORS["text_muted"], "fontSize": "11px", "padding": "8px 0", "fontWeight": "400"}),
                        html.Th("Rank", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px", "fontWeight": "400"}),
                        html.Th("Value", style={"textAlign": "right", "color": COLORS["text_muted"], "fontSize": "11px", "fontWeight": "400"}),
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Points Allowed", style={"padding": "10px 0", "fontSize": "13px", "fontWeight": "500"}),
                        html.Td(f"#{pts_rank}", style={"textAlign": "center", "color": get_rank_color(pts_rank), "fontSize": "14px", "fontWeight": "600"}),
                        html.Td(f"{pts_allowed:.1f}", style={"textAlign": "right", "fontSize": "13px"}),
                    ]),
                    html.Tr([
                        html.Td("Assists Allowed", style={"padding": "10px 0", "fontSize": "13px", "fontWeight": "500"}),
                        html.Td(f"#{ast_rank}", style={"textAlign": "center", "color": get_rank_color(ast_rank), "fontSize": "14px", "fontWeight": "600"}),
                        html.Td(f"{ast_allowed:.1f}", style={"textAlign": "right", "fontSize": "13px"}),
                    ]),
                    html.Tr([
                        html.Td("Rebounds Allowed", style={"padding": "10px 0", "fontSize": "13px", "fontWeight": "500"}),
                        html.Td(f"#{reb_rank}", style={"textAlign": "center", "color": get_rank_color(reb_rank), "fontSize": "14px", "fontWeight": "600"}),
                        html.Td(f"{reb_allowed:.1f}", style={"textAlign": "right", "fontSize": "13px"}),
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
    """Create team rankings comparison table"""
    # Get player's team
    last_game = player_df.sort_values("_date", ascending=False).iloc[0]
    matchup = last_game.get("MATCHUP", "")
    player_team = matchup.split()[0] if isinstance(matchup, str) and matchup else "UNK"

    # Get team stats
    player_team_stats = TEAM_DEF[TEAM_DEF["TEAM_ABBREVIATION"] == player_team] if "TEAM_ABBREVIATION" in TEAM_DEF.columns else pd.DataFrame()
    opp_team_stats = TEAM_DEF[TEAM_DEF["TEAM_ABBREVIATION"] == opponent] if "TEAM_ABBREVIATION" in TEAM_DEF.columns else pd.DataFrame()

    stats_to_show = [
        ("Points", "OPP_PTS", "PTS"),
        ("Field Goal Pct", "OPP_FG_PCT", "FG_PCT"),
        ("3-Point Pct", "OPP_FG3_PCT", "FG3_PCT"),
        ("Rebounds", "OPP_REB", "REB"),
    ]

    rows = []
    for stat_name, def_col, off_col in stats_to_show:
        # Player team values
        pt_val = 0
        pt_rank = "-"
        if not player_team_stats.empty:
            pt_val = player_team_stats.iloc[0].get(off_col, 0)
            if off_col in TEAM_DEF.columns:
                pt_rank = int(TEAM_DEF[off_col].rank(ascending=False).get(player_team_stats.index[0], 15))

        # Opponent team values
        opp_val = 0
        opp_rank = "-"
        if not opp_team_stats.empty:
            opp_val = opp_team_stats.iloc[0].get(def_col, opp_team_stats.iloc[0].get(off_col, 0))
            if def_col in TEAM_DEF.columns:
                opp_rank = int(TEAM_DEF[def_col].rank(ascending=False).get(opp_team_stats.index[0], 15))

        # Format values
        if "Pct" in stat_name:
            pt_display = f"{pt_val * 100:.1f}" if pt_val < 1 else f"{pt_val:.1f}"
            opp_display = f"{opp_val * 100:.1f}" if opp_val < 1 else f"{opp_val:.1f}"
        else:
            pt_display = f"{pt_val:.1f}"
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

        if status_text in ["UNKNOWN", "HEALTHY"]:
            status_text = "ACTIVE"
    except Exception:
        status_text = "ACTIVE"
        reason = ""
        news = []
        injury_type = ""

    # Status colors and labels
    status_config = {
        "ACTIVE": {"color": COLORS["hit_high"], "label": "ACTIVE", "icon": "✓"},
        "PROBABLE": {"color": COLORS["hit_high"], "label": "PROBABLE", "icon": "●"},
        "QUESTIONABLE": {"color": COLORS["hit_mid"], "label": "QUESTIONABLE", "icon": "?"},
        "DOUBTFUL": {"color": COLORS["hit_low"], "label": "DOUBTFUL", "icon": "!"},
        "OUT": {"color": COLORS["hit_low"], "label": "OUT", "icon": "✕"},
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
            html.Div(news_elements)
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

    # If no game today, use most recent opponent
    if not opponent:
        last_matchup = player_df.iloc[0].get("MATCHUP", "")
        if isinstance(last_matchup, str):
            if "@" in last_matchup:
                opponent = last_matchup.split("@")[-1].strip()[:3]
            elif "vs." in last_matchup:
                opponent = last_matchup.split("vs.")[-1].strip()[:3]

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
    # Get model prediction
    model_prediction = l5_avg
    if "PTS" in PREDICTORS:
        try:
            result = PREDICTORS["PTS"].predict_player_game(player_name, DF)
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
    Create the Best Props tab content showing today's top picks.
    Includes single stats (PTS, AST, REB) and combos (PTS+AST, PTS+REB, PRA).
    Shows WHY each prop was selected with matchup analysis.
    """
    # Get today's actual games
    today_games = get_todays_games()

    if today_games.empty:
        return html.Div([
            html.Div([
                html.Div("BEST PROPS", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "letterSpacing": "1px",
                    "marginBottom": "16px"
                }),
                html.Div("No games scheduled today", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "14px",
                    "textAlign": "center",
                    "padding": "40px 0"
                }),
            ], style=CARD)
        ])

    # Build team -> opponent mapping from TODAY's games
    team_to_opponent = {}
    team_is_home = {}
    for _, game in today_games.iterrows():
        home = game.get("HOME_TEAM", "")
        away = game.get("AWAY_TEAM", "")
        if home and away:
            team_to_opponent[home] = away
            team_to_opponent[away] = home
            team_is_home[home] = True
            team_is_home[away] = False

    teams_today = set(team_to_opponent.keys())

    if not teams_today:
        return html.Div([
            html.Div([
                html.Div("BEST PROPS", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "letterSpacing": "1px",
                    "marginBottom": "16px"
                }),
                html.Div("No games scheduled today", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "14px",
                    "textAlign": "center",
                    "padding": "40px 0"
                }),
            ], style=CARD)
        ])

    # Get players on teams playing today (skip injured players)
    players_today = []
    player_info = {}  # Store player info for analysis

    for player_name in PLAYERS:
        # Must have game history
        player_df = DF[DF["PLAYER_NAME"] == player_name]
        if len(player_df) == 0:
            continue

        # Get player's CURRENT team from PLAYER_POSITIONS (most accurate source)
        player_team = ""
        position = "F"
        if not PLAYER_POSITIONS.empty:
            pos_match = PLAYER_POSITIONS[PLAYER_POSITIONS["PLAYER_NAME"] == player_name]
            if len(pos_match) > 0:
                player_team = str(pos_match["TEAM_ABBREVIATION"].iloc[0])
                position = str(pos_match["POSITION"].iloc[0])

        # Skip if we couldn't find the player's current team
        if not player_team:
            continue

        # Check if player's team is playing today
        if player_team in teams_today:
            # Check if player is injured (OUT status)
            try:
                injury_status = get_player_injury_status(player_name)
                if injury_status.get("status") == "OUT":
                    continue  # Skip injured players
            except Exception:
                pass  # If we can't check, include the player

            # Get TODAY's actual opponent
            opponent = team_to_opponent.get(player_team, "")

            players_today.append(player_name)
            player_info[player_name] = {
                "team": player_team,
                "opponent": opponent,
                "position": position,
                "is_home": team_is_home.get(player_team, False)
                }

    if not players_today:
        players_today = PLAYERS[:50]

    # Define prop types to analyze (singles and combos)
    prop_types = [
        {"name": "PTS", "stats": ["PTS"], "label": "Points"},
        {"name": "AST", "stats": ["AST"], "label": "Assists"},
        {"name": "REB", "stats": ["REB"], "label": "Rebounds"},
        {"name": "PTS+AST", "stats": ["PTS", "AST"], "label": "Pts+Ast"},
        {"name": "PTS+REB", "stats": ["PTS", "REB"], "label": "Pts+Reb"},
        {"name": "AST+REB", "stats": ["AST", "REB"], "label": "Ast+Reb"},
        {"name": "PRA", "stats": ["PTS", "AST", "REB"], "label": "Pts+Ast+Reb"},
        {"name": "3PM", "stats": ["FG3M"], "label": "3-Pointers"},
    ]

    # Generate all props with analysis
    all_props = []

    for player_name in players_today[:40]:  # Limit for performance
        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)
        recent = player_df.head(10)
        info = player_info.get(player_name, {})

        if len(recent) < 5:
            continue

        for prop_type in prop_types:
            try:
                # Calculate combined stat value
                stat_cols = [s for s in prop_type["stats"] if s in recent.columns]
                if len(stat_cols) != len(prop_type["stats"]):
                    continue

                recent_vals = recent[stat_cols].sum(axis=1)
                season_vals = player_df[stat_cols].sum(axis=1)

                l5_avg = recent_vals.head(5).mean()
                l10_avg = recent_vals.mean()
                season_avg = season_vals.mean()
                std_dev = recent_vals.std() if len(recent_vals) > 1 else l10_avg * 0.2

                # Set line based on recent average (round to 0.5)
                line = round(l10_avg * 2) / 2

                # Get model prediction for single stats
                prediction = l10_avg
                if len(prop_type["stats"]) == 1 and prop_type["stats"][0].upper() in PREDICTORS:
                    try:
                        result = PREDICTORS[prop_type["stats"][0].upper()].predict_player_game(player_name, DF)
                        if "error" not in result:
                            pred_key = f"predicted_{prop_type['stats'][0].lower()}"
                            prediction = result.get(pred_key, l10_avg)
                    except Exception:
                        pass
                elif len(prop_type["stats"]) > 1:
                    # For combos, sum predictions
                    combo_pred = 0
                    for stat in prop_type["stats"]:
                        if stat.upper() in PREDICTORS:
                            try:
                                result = PREDICTORS[stat.upper()].predict_player_game(player_name, DF)
                                if "error" not in result:
                                    combo_pred += result.get(f"predicted_{stat.lower()}", 0)
                            except Exception:
                                combo_pred += recent[stat].mean()
                        else:
                            combo_pred += recent[stat].mean() if stat in recent.columns else 0
                    prediction = combo_pred if combo_pred > 0 else l10_avg

                # Calculate hit probability
                hit_prob = calculate_hit_probability(prediction, line, std_dev, "over")
                l10_rate = (recent_vals >= line).sum() / len(recent_vals)
                l5_rate = (recent_vals.head(5) >= line).sum() / 5

                # Analyze matchup favorability
                matchup_edge = 0
                matchup_reason = ""

                if info.get("opponent") and not DEFENSE_VS_POS.empty:
                    position = info.get("position", "F")
                    opp = info.get("opponent", "")

                    # Get opponent defense vs this position
                    def_data = DEFENSE_VS_POS[
                        (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opp) &
                        (DEFENSE_VS_POS["POSITION"] == position)
                    ]

                    if len(def_data) > 0:
                        def_row = def_data.iloc[0]
                        # Check relevant defensive ranking
                        if "PTS" in prop_type["stats"]:
                            pts_rank = def_row.get("PTS_RANK", 15)
                            if pts_rank <= 10:
                                matchup_edge += 0.1
                                matchup_reason = f"{opp} allows most PTS vs {position}s (#{int(pts_rank)})"
                        if "AST" in prop_type["stats"]:
                            ast_rank = def_row.get("AST_RANK", 15)
                            if ast_rank <= 10:
                                matchup_edge += 0.08
                                if matchup_reason:
                                    matchup_reason += f", AST #{int(ast_rank)}"
                                else:
                                    matchup_reason = f"{opp} allows high AST vs {position}s (#{int(ast_rank)})"
                        if "REB" in prop_type["stats"]:
                            reb_rank = def_row.get("REB_RANK", 15)
                            if reb_rank <= 10:
                                matchup_edge += 0.05
                                if not matchup_reason:
                                    matchup_reason = f"{opp} allows high REB vs {position}s (#{int(reb_rank)})"

                # Build reasoning
                reasons = []

                # Trend analysis
                if l5_avg > season_avg * 1.1:
                    reasons.append(f"🔥 Hot streak: L5 avg {l5_avg:.1f} > season {season_avg:.1f}")
                elif l5_avg > l10_avg:
                    reasons.append(f"📈 Trending up: L5 {l5_avg:.1f} vs L10 {l10_avg:.1f}")

                # Hit rate analysis
                if l10_rate >= 0.7:
                    reasons.append(f"✓ Hit {int(l10_rate*100)}% of L10 games over {line:.1f}")
                elif l5_rate >= 0.8:
                    reasons.append(f"✓ Hit {int(l5_rate*100)}% of L5 games over {line:.1f}")

                # Matchup analysis
                if matchup_reason:
                    reasons.append(f"🎯 {matchup_reason}")

                # Home/away boost
                if info.get("is_home"):
                    home_vals = player_df[player_df["is_home"] == 1][stat_cols].sum(axis=1)
                    if len(home_vals) > 3 and home_vals.mean() > l10_avg * 1.05:
                        reasons.append(f"🏠 Home boost: avg {home_vals.mean():.1f} at home")

                # Only include if we have good reasons and probability
                adjusted_prob = min(hit_prob + matchup_edge, 0.95)

                if adjusted_prob >= 0.55 and len(reasons) >= 1:
                    all_props.append({
                        "player": player_name,
                        "prop_type": prop_type["name"],
                        "prop_label": prop_type["label"],
                        "prediction": prediction,
                        "line": line,
                        "hit_prob": adjusted_prob,
                        "l10_rate": l10_rate,
                        "l5_rate": l5_rate,
                        "reasons": reasons,
                        "matchup_edge": matchup_edge,
                        "opponent": info.get("opponent", ""),
                        "is_home": info.get("is_home", False)
                    })

            except Exception:
                continue

    # Sort by hit probability and matchup edge
    all_props.sort(key=lambda x: (x["hit_prob"] + x["matchup_edge"] * 0.5), reverse=True)

    # Take top props, ensuring variety
    final_props = []
    seen_players = {}
    for prop in all_props:
        player = prop["player"]
        if seen_players.get(player, 0) < 2:  # Max 2 props per player
            final_props.append(prop)
            seen_players[player] = seen_players.get(player, 0) + 1
        if len(final_props) >= 15:
            break

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

    # Build prop cards with explanations
    prop_cards = []
    for prop in final_props:
        prob_color = get_hit_color(prop["hit_prob"] * 100)

        # Build reasons list
        reason_elements = [
            html.Div(reason, style={
                "fontSize": "11px",
                "color": COLORS["text_secondary"],
                "marginBottom": "4px",
                "paddingLeft": "8px",
                "borderLeft": f"2px solid {COLORS['border']}"
            }) for reason in prop["reasons"][:3]
        ]

        # Build opponent span
        opponent_span = html.Span(f" vs {prop['opponent']}", style={
            "color": COLORS["text_muted"],
            "fontSize": "12px",
            "marginLeft": "4px"
        }) if prop["opponent"] else None

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
                        html.Span(" 🏠" if prop["is_home"] else " ✈️", style={
                            "fontSize": "12px",
                            "marginLeft": "4px"
                        })
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
                        })
                    ], style={"marginTop": "2px"})
                ], style={"flex": "1"}),

                # Right: Probability badge
                html.Div([
                    html.Div(f"{prop['hit_prob']*100:.0f}%", style={
                        "color": prob_color,
                        "fontWeight": "700",
                        "fontSize": "18px"
                    }),
                    html.Div(f"L10: {prop['l10_rate']*100:.0f}%", style={
                        "color": COLORS["text_muted"],
                        "fontSize": "10px"
                    })
                ], style={"textAlign": "right"})
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "8px"
            }),

            # WHY section - reasons for selection
            html.Div(reason_elements, style={"marginTop": "8px"})

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
        ], style={**CARD, "marginBottom": "12px"}),

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
    print("🏀 NBA Props Dashboard")
    print("=" * 50)
    print("\nOpen: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop\n")

    app.run(debug=True, port=8050)
