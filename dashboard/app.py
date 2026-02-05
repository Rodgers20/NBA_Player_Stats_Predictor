# dashboard/app.py
"""
NBA Player Props Dashboard - Outlier.bet Style
=============================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, callback
import plotly.graph_objects as go

from utils.feature_engineering import engineer_features
from utils.injury_news import get_player_injury_status
from models.predictor import NBAPredictor

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    game_logs = pd.read_csv(os.path.join(data_dir, "player_game_logs.csv"))
    team_def = pd.read_csv(os.path.join(data_dir, "team_defensive_stats.csv"))
    df = engineer_features(game_logs, team_def)
    df["_date"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df = df.sort_values(["PLAYER_NAME", "_date"])
    return df, team_def


def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    predictors = {}
    for target in ["pts", "ast", "reb"]:
        filepath = os.path.join(models_dir, f"{target}_predictor.pkl")
        if os.path.exists(filepath):
            predictors[target.upper()] = NBAPredictor.load(filepath)
    return predictors


print("Loading data...")
DF, TEAM_DEF = load_data()
PREDICTORS = load_models()
PLAYERS = sorted(DF["PLAYER_NAME"].unique().tolist())

# Build player ID mapping from the data
PLAYER_IDS = {}
if "PLAYER_ID" in DF.columns:
    for name in PLAYERS:
        pid = DF[DF["PLAYER_NAME"] == name]["PLAYER_ID"].iloc[0] if len(DF[DF["PLAYER_NAME"] == name]) > 0 else None
        if pid:
            PLAYER_IDS[name] = int(pid)

print(f"Loaded {len(DF)} game records for {len(PLAYERS)} players")

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

# Stat type configurations
STAT_TYPES = [
    {"id": "REB+AST", "label": "REB+AST"},
    {"id": "PTS", "label": "PTS"},
    {"id": "AST", "label": "AST"},
    {"id": "PTS+AST", "label": "PTS+AST"},
    {"id": "FG3M", "label": "3PTM"},
    {"id": "BLK", "label": "BLK"},
    {"id": "STL", "label": "STL"},
    {"id": "STL+BLK", "label": "STL+BLK"},
    {"id": "REB", "label": "REB"},
    {"id": "PTS+REB", "label": "PTS+REB"},
    {"id": "PTS+AST+REB", "label": "PRA"},
]

# =============================================================================
# DASH APP
# =============================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# =============================================================================
# STYLES
# =============================================================================

CARD = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "16px",
    "border": f"1px solid {COLORS['border']}"
}

TAB_STYLE = {
    "padding": "8px 14px",
    "borderRadius": "6px",
    "cursor": "pointer",
    "fontSize": "13px",
    "fontWeight": "500",
    "color": COLORS["text_secondary"],
    "backgroundColor": "transparent",
    "border": "none",
    "marginRight": "4px",
    "whiteSpace": "nowrap",
}

TAB_ACTIVE = {
    **TAB_STYLE,
    "backgroundColor": COLORS["border"],
    "color": COLORS["text"],
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
                    style={"width": "300px", "backgroundColor": COLORS["card"]}
                ),
                html.Div(id="player-header", style={"marginTop": "8px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
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
                        style=TAB_ACTIVE if s["id"] == "REB+AST" else TAB_STYLE
                    ) for s in STAT_TYPES
                ], style={"display": "flex", "flexWrap": "nowrap"})
            ], style={
                "overflowX": "auto",
                "marginBottom": "16px",
                "paddingBottom": "8px",
            }),

            # Store for selected stat
            dcc.Store(id="selected-stat", data="REB+AST"),

            # Time period tabs
            html.Div([
                html.Button("L5", id="period-l5", n_clicks=0, style=TAB_STYLE),
                html.Button("L10", id="period-l10", n_clicks=0, style=TAB_ACTIVE),
                html.Button("L20", id="period-l20", n_clicks=0, style=TAB_STYLE),
                html.Button("2024", id="period-2024", n_clicks=0, style=TAB_STYLE),
                html.Button("2023", id="period-2023", n_clicks=0, style=TAB_STYLE),
            ], style={"display": "flex", "marginBottom": "20px"}),

            dcc.Store(id="selected-period", data=10),
            dcc.Store(id="selected-season", data=None),

            # Hit rate header with avg/median
            html.Div(id="hit-rate-header", style={"marginBottom": "20px"}),

            # Threshold slider
            html.Div([
                html.Div("Threshold:", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "12px",
                    "marginRight": "12px"
                }),
                dcc.Slider(
                    id="threshold-slider",
                    min=0,
                    max=50,
                    step=0.5,
                    value=10,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
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

        ], style={"flex": "2", "marginRight": "20px"}),

        # Right panel - Sidebar
        html.Div([
            # Tabs for sidebar sections
            html.Div([
                html.Button("Matchup", id="sidebar-matchup", n_clicks=1, style=TAB_ACTIVE),
                html.Button("Injuries", id="sidebar-injuries", n_clicks=0, style=TAB_STYLE),
                html.Button("Insights", id="sidebar-insights", n_clicks=0, style=TAB_STYLE),
            ], style={"display": "flex", "marginBottom": "16px"}),

            dcc.Store(id="sidebar-tab", data="matchup"),

            # Dynamic sidebar content
            html.Div(id="sidebar-content"),

        ], style={"width": "380px", "flexShrink": "0"}),

    ], style={
        "display": "flex",
        "padding": "24px",
        "maxWidth": "1600px",
        "margin": "0 auto"
    }),

], style={
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "color": COLORS["text"]
})


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
        return "REB+AST"

    for s in STAT_TYPES:
        tab_id = f"tab-{s['id'].lower().replace('+', '-')}"
        if triggered == tab_id:
            return s["id"]

    return "REB+AST"


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
     Output("period-l5", "style"),
     Output("period-l10", "style"),
     Output("period-l20", "style"),
     Output("period-2024", "style"),
     Output("period-2023", "style")],
    [Input("period-l5", "n_clicks"),
     Input("period-l10", "n_clicks"),
     Input("period-l20", "n_clicks"),
     Input("period-2024", "n_clicks"),
     Input("period-2023", "n_clicks")]
)
def update_period_tabs(l5, l10, l20, y2024, y2023):
    from dash import ctx
    triggered = ctx.triggered_id

    styles = [TAB_STYLE] * 5
    period = 10
    season = None

    if triggered == "period-l5":
        period, styles[0] = 5, TAB_ACTIVE
    elif triggered == "period-l10" or triggered is None:
        period, styles[1] = 10, TAB_ACTIVE
    elif triggered == "period-l20":
        period, styles[2] = 20, TAB_ACTIVE
    elif triggered == "period-2024":
        period, season, styles[3] = 100, "2024-25", TAB_ACTIVE
    elif triggered == "period-2023":
        period, season, styles[4] = 100, "2023-24", TAB_ACTIVE

    return [period, season] + styles


# Sidebar tab handler
@callback(
    [Output("sidebar-tab", "data"),
     Output("sidebar-matchup", "style"),
     Output("sidebar-injuries", "style"),
     Output("sidebar-insights", "style")],
    [Input("sidebar-matchup", "n_clicks"),
     Input("sidebar-injuries", "n_clicks"),
     Input("sidebar-insights", "n_clicks")]
)
def update_sidebar_tabs(matchup, injuries, insights):
    from dash import ctx
    triggered = ctx.triggered_id

    styles = [TAB_STYLE] * 3
    tab = "matchup"

    if triggered == "sidebar-matchup" or triggered is None:
        tab, styles[0] = "matchup", TAB_ACTIVE
    elif triggered == "sidebar-injuries":
        tab, styles[1] = "injuries", TAB_ACTIVE
    elif triggered == "sidebar-insights":
        tab, styles[2] = "insights", TAB_ACTIVE

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
    current = player_df[player_df["SEASON"] == "2024-25"]
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
    Output("hit-rate-header", "children"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data"),
     Input("selected-period", "data"),
     Input("threshold-slider", "value")]
)
def update_hit_rate_header(player_name, stat, period, threshold):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

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

    hit_pct, hits = calc_hit(player_df, period)
    l5_pct, _ = calc_hit(player_df, 5)
    l20_pct, _ = calc_hit(player_df, 20)

    # Season hit rates by year
    df_2024 = player_df[player_df["SEASON"] == "2024-25"]
    df_2023 = player_df[player_df["SEASON"] == "2023-24"]
    pct_2024, _ = calc_hit(df_2024, len(df_2024)) if len(df_2024) > 0 else (0, 0)
    pct_2023, _ = calc_hit(df_2023, len(df_2023)) if len(df_2023) > 0 else (0, 0)

    # Get display name for stat
    stat_display = stat.replace("+", " + ")
    for s in STAT_TYPES:
        if s["id"] == stat:
            stat_display = s["label"]
            break

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
                html.Div("2024", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{pct_2024:.0f}%", style={"color": get_hit_color(pct_2024), "fontWeight": "600"})
            ], style={"textAlign": "center", "marginRight": "16px"}),

            html.Div([
                html.Div("2023", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{pct_2023:.0f}%", style={"color": get_hit_color(pct_2023), "fontWeight": "600"})
            ], style={"textAlign": "center"}),

        ], style={"display": "flex", "alignItems": "flex-end"}),
    ])


@callback(
    Output("avg-median-footer", "children"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data"),
     Input("selected-period", "data")]
)
def update_avg_median(player_name, stat, period):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False).head(period)

    if "+" in stat:
        parts = stat.split("+")
        vals = sum(player_df[p] for p in parts if p in player_df.columns)
    else:
        vals = player_df[stat] if stat in player_df.columns else pd.Series([0])

    avg = vals.mean()
    median = vals.median()

    return [
        html.Div([
            html.Span("Average", style={"color": COLORS["text_secondary"], "marginRight": "8px"}),
            html.Span(f"{avg:.1f}", style={"fontWeight": "600"})
        ]),
        html.Div([
            html.Span("Median", style={"color": COLORS["text_secondary"], "marginRight": "8px"}),
            html.Span(f"{median:.1f}", style={"fontWeight": "600"})
        ])
    ]


@callback(
    Output("main-chart", "figure"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data"),
     Input("selected-period", "data"),
     Input("selected-season", "data"),
     Input("threshold-slider", "value")]
)
def update_main_chart(player_name, stat, period, season, threshold):
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # Filter by season if specified
    if season:
        player_df = player_df[player_df["SEASON"] == season]

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
     Input("selected-stat", "data")]
)
def update_sidebar_content(tab, player_name, stat):
    if not player_name:
        return None

    if tab == "matchup":
        return create_matchup_content(player_name, stat)
    elif tab == "injuries":
        return create_injuries_content(player_name)
    else:
        return create_insights_content(player_name, stat)


def create_matchup_content(player_name, stat):
    """Create the matchup analysis sidebar content"""
    player_df = DF[DF["PLAYER_NAME"] == player_name]

    # Get stat display name
    stat_display = stat.replace("+", " + ")

    # Get unique opponents and their defensive stats
    if "MATCHUP" in player_df.columns:
        opponents = []
        for matchup in player_df["MATCHUP"].unique():
            if isinstance(matchup, str):
                if "@" in matchup:
                    opp = matchup.split("@")[-1].strip()[:3]
                elif "vs." in matchup:
                    opp = matchup.split("vs.")[-1].strip()[:3]
                else:
                    continue
                opponents.append(opp)
        opponents = list(set(opponents))[:10]
    else:
        opponents = []

    # Get team defensive stats
    def_stats = []
    for _, row in TEAM_DEF.iterrows():
        team = row.get("TEAM_ABBREVIATION", "")
        pts_allowed = row.get("PTS", 0)
        ast_allowed = row.get("AST", 0)
        reb_allowed = row.get("REB", 0)
        def_stats.append({
            "team": team,
            "pts": pts_allowed,
            "ast": ast_allowed,
            "reb": reb_allowed,
            "rank": row.get("RANK", 0) if "RANK" in row else 0
        })

    # Sort by relevant stat
    primary_stat = stat.split("+")[0] if "+" in stat else stat
    def_stats.sort(key=lambda x: x.get(primary_stat.lower(), 0), reverse=True)

    return html.Div([
        # Defense section
        html.Div([
            html.Div(f"League Defense vs {stat_display}", style={
                "color": COLORS["text"],
                "fontSize": "14px",
                "fontWeight": "600",
                "marginBottom": "16px"
            }),

            # Team defensive rankings table
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Team", style={"textAlign": "left", "color": COLORS["text_muted"], "fontSize": "11px", "padding": "8px 0"}),
                        html.Th("Rank", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
                        html.Th("Allowed", style={"textAlign": "right", "color": COLORS["text_muted"], "fontSize": "11px"}),
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(d["team"], style={"padding": "6px 0", "fontSize": "12px"}),
                        html.Td(f"{i+1}", style={"textAlign": "center", "color": COLORS["hit_low"], "fontSize": "12px"}),
                        html.Td(f"{d.get(primary_stat.lower(), 0):.1f}", style={"textAlign": "right", "fontSize": "12px"}),
                    ]) for i, d in enumerate(def_stats[:10])
                ])
            ], style={"width": "100%", "borderCollapse": "collapse"})
        ], style=CARD),

        # Splits section
        html.Div([
            html.Div("SPLITS", style={
                "color": COLORS["text_secondary"],
                "fontSize": "11px",
                "fontWeight": "600",
                "letterSpacing": "1px",
                "marginBottom": "12px"
            }),
            create_splits_table(player_name)
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


def create_injuries_content(player_name):
    """Create the injuries tab content"""
    try:
        status = get_player_injury_status(player_name)
        status_text = status.get("status", "ACTIVE")
        reason = status.get("reason", "")
        news = status.get("news", [])

        if status_text in ["UNKNOWN", "HEALTHY"]:
            status_text = "ACTIVE"
    except:
        status_text = "ACTIVE"
        reason = ""
        news = []

    status_color = {
        "ACTIVE": COLORS["hit_high"],
        "QUESTIONABLE": COLORS["hit_mid"],
        "OUT": COLORS["hit_low"],
        "DOUBTFUL": COLORS["hit_low"],
    }.get(status_text, COLORS["hit_high"])

    player_df = DF[DF["PLAYER_NAME"] == player_name]
    games_24 = len(player_df[player_df["SEASON"] == "2024-25"])

    return html.Div([
        html.Div([
            html.Div("INJURY STATUS", style={
                "color": COLORS["text_secondary"],
                "fontSize": "11px",
                "fontWeight": "600",
                "letterSpacing": "1px",
                "marginBottom": "16px"
            }),
            html.Div([
                html.Span("●", style={"color": status_color, "marginRight": "8px", "fontSize": "16px"}),
                html.Span(status_text, style={"color": status_color, "fontWeight": "600", "fontSize": "18px"})
            ]),
            html.Div(reason, style={
                "color": COLORS["text_secondary"],
                "fontSize": "13px",
                "marginTop": "8px"
            }) if reason else None,
            html.Div(f"{games_24} games played in 2024-25", style={
                "color": COLORS["text_muted"],
                "fontSize": "12px",
                "marginTop": "12px"
            }),
        ], style=CARD),

        # Recent news
        html.Div([
            html.Div("RECENT NEWS", style={
                "color": COLORS["text_secondary"],
                "fontSize": "11px",
                "fontWeight": "600",
                "letterSpacing": "1px",
                "marginBottom": "12px"
            }),
            html.Div([
                html.Div([
                    html.Div(n.get("title", "No recent news"), style={
                        "fontSize": "13px",
                        "marginBottom": "4px"
                    }),
                    html.Div(n.get("date", ""), style={
                        "fontSize": "11px",
                        "color": COLORS["text_muted"]
                    })
                ], style={"marginBottom": "12px", "paddingBottom": "12px", "borderBottom": f"1px solid {COLORS['border']}"})
                for n in (news[:3] if news else [{"title": "No recent news available"}])
            ])
        ], style=CARD),
    ])


def create_insights_content(player_name, stat):
    """Create the insights tab content"""
    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # Calculate trends
    l5_avg = player_df.head(5)["PTS"].mean() if len(player_df) >= 5 else 0
    l20_avg = player_df.head(20)["PTS"].mean() if len(player_df) >= 20 else 0
    season_avg = player_df["PTS"].mean()

    trend = "up" if l5_avg > season_avg else "down" if l5_avg < season_avg else "flat"
    trend_color = COLORS["hit_high"] if trend == "up" else COLORS["hit_low"] if trend == "down" else COLORS["text_secondary"]
    trend_icon = "↑" if trend == "up" else "↓" if trend == "down" else "→"

    # Get predictions
    predictions = []
    for stat_key, predictor in PREDICTORS.items():
        try:
            result = predictor.predict_player_game(player_name, DF)
            if "error" not in result:
                pred_key = f"predicted_{stat_key.lower()}"
                predictions.append({
                    "stat": stat_key,
                    "predicted": result.get(pred_key, 0),
                    "avg": result.get("recent_avg", 0)
                })
        except:
            pass

    return html.Div([
        # Trend card
        html.Div([
            html.Div("RECENT TREND", style={
                "color": COLORS["text_secondary"],
                "fontSize": "11px",
                "fontWeight": "600",
                "letterSpacing": "1px",
                "marginBottom": "12px"
            }),
            html.Div([
                html.Span(trend_icon, style={"color": trend_color, "fontSize": "24px", "marginRight": "12px"}),
                html.Span(f"{'Trending up' if trend == 'up' else 'Trending down' if trend == 'down' else 'Consistent'}", style={
                    "color": trend_color,
                    "fontWeight": "600"
                })
            ]),
            html.Div([
                html.Div(f"L5 Avg: {l5_avg:.1f}", style={"fontSize": "13px"}),
                html.Div(f"Season Avg: {season_avg:.1f}", style={"fontSize": "13px", "color": COLORS["text_secondary"]}),
            ], style={"marginTop": "12px"})
        ], style=CARD),

        # Predictions card
        html.Div([
            html.Div("ML PROJECTIONS", style={
                "color": COLORS["text_secondary"],
                "fontSize": "11px",
                "fontWeight": "600",
                "letterSpacing": "1px",
                "marginBottom": "16px"
            }),
            html.Div([
                html.Div([
                    html.Div(pred["stat"], style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginBottom": "4px"}),
                    html.Div(f"{pred['predicted']:.1f}", style={
                        "color": get_stat_color(pred["stat"]),
                        "fontSize": "24px",
                        "fontWeight": "600"
                    }),
                    html.Div(f"avg {pred['avg']:.1f}", style={"color": COLORS["text_muted"], "fontSize": "11px"})
                ], style={
                    "textAlign": "center",
                    "padding": "12px",
                    "backgroundColor": COLORS["bg"],
                    "borderRadius": "8px",
                    "flex": "1",
                    "margin": "0 4px"
                }) for pred in predictions
            ], style={"display": "flex"})
        ], style=CARD),
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
