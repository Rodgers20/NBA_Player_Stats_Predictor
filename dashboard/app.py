# dashboard/app.py
"""
NBA Player Stats Dashboard
==========================
An interactive dashboard for analyzing NBA player performance
and making stat predictions.

HOW DASH WORKS:
---------------
Dash is a Python framework for building web dashboards. It combines:
1. Plotly - for interactive charts
2. HTML/CSS - for layout (but written in Python!)
3. Callbacks - for interactivity (when user does X, do Y)

STRUCTURE:
- app.layout = The HTML structure (what users see)
- @app.callback = Functions that update the page when users interact

TO RUN:
    python dashboard/app.py
    Then open http://127.0.0.1:8050 in your browser
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
from utils.feature_engineering import engineer_features
from utils.injury_news import get_player_injury_status
from models.predictor import NBAPredictor

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare all data for the dashboard."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # Load game logs
    game_logs = pd.read_csv(os.path.join(data_dir, "player_game_logs.csv"))

    # Load team defensive stats
    team_def = pd.read_csv(os.path.join(data_dir, "team_defensive_stats.csv"))

    # Engineer features
    df = engineer_features(game_logs, team_def)

    # Parse dates for proper sorting
    df["_date"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df = df.sort_values(["PLAYER_NAME", "_date"])

    return df


def load_models():
    """Load trained prediction models."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    predictors = {}

    for target in ["pts", "ast", "reb"]:
        filepath = os.path.join(models_dir, f"{target}_predictor.pkl")
        if os.path.exists(filepath):
            predictors[target.upper()] = NBAPredictor.load(filepath)

    return predictors


# Load data at startup
print("Loading data...")
DF = load_data()
PREDICTORS = load_models()
PLAYERS = sorted(DF["PLAYER_NAME"].unique().tolist())

print(f"Loaded {len(DF)} game records for {len(PLAYERS)} players")

# =============================================================================
# DASH APP SETUP
# =============================================================================

# Create the Dash app
# external_stylesheets adds Bootstrap-like styling for nicer appearance
app = Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/darkly/bootstrap.min.css"
    ],
    suppress_callback_exceptions=True
)

# For deployment
server = app.server

# =============================================================================
# COLOR SCHEME
# =============================================================================

COLORS = {
    "background": "#222222",
    "card_bg": "#2d2d2d",
    "text": "#ffffff",
    "accent": "#00bc8c",  # Teal/green accent
    "secondary": "#3498db",  # Blue
    "warning": "#f39c12",  # Orange
    "danger": "#e74c3c",  # Red
}

# =============================================================================
# LAYOUT COMPONENTS
# =============================================================================

def create_header():
    """Create the dashboard header."""
    return html.Div([
        html.H1("🏀 NBA Player Stats Predictor",
                style={"color": COLORS["accent"], "marginBottom": "5px"}),
        html.P("Analyze performance trends and predict player stats",
               style={"color": "#888", "marginTop": "0"})
    ], style={"textAlign": "center", "padding": "20px"})


def create_player_selector():
    """Create the player search/selection dropdown."""
    return html.Div([
        html.Label("Select Player:", style={"fontWeight": "bold", "color": COLORS["text"]}),
        dcc.Dropdown(
            id="player-dropdown",
            options=[{"label": p, "value": p} for p in PLAYERS],
            value=PLAYERS[0] if PLAYERS else None,
            placeholder="Search for a player...",
            style={"color": "#000"},  # Dark text for dropdown
            searchable=True,
            clearable=False
        )
    ], style={
        "width": "400px",
        "margin": "0 auto",
        "padding": "20px"
    })


def create_stats_cards():
    """Create the summary stat cards."""
    return html.Div([
        html.Div(id="stats-cards", children=[
            # Cards will be populated by callback
        ])
    ], style={"padding": "10px"})


def create_main_content():
    """Create the main dashboard content area."""
    return html.Div([
        # Row 1: Stats cards
        html.Div(id="stats-cards-container", style={"marginBottom": "20px"}),

        # Row 2: Charts
        html.Div([
            # Left: Season Trend Chart
            html.Div([
                html.H4("Season Performance Trend", style={"color": COLORS["text"]}),
                dcc.Graph(id="season-trend-chart")
            ], style={
                "flex": "2",
                "backgroundColor": COLORS["card_bg"],
                "borderRadius": "10px",
                "padding": "15px",
                "marginRight": "10px"
            }),

            # Right: Prediction Panel
            html.Div([
                html.H4("Prediction", style={"color": COLORS["text"]}),
                html.Div(id="prediction-panel")
            ], style={
                "flex": "1",
                "backgroundColor": COLORS["card_bg"],
                "borderRadius": "10px",
                "padding": "15px"
            })
        ], style={"display": "flex", "marginBottom": "20px"}),

        # Row 3: Home/Away and Matchup Analysis
        html.Div([
            # Left: Home vs Away
            html.Div([
                html.H4("Home vs Away", style={"color": COLORS["text"]}),
                dcc.Graph(id="home-away-chart")
            ], style={
                "flex": "1",
                "backgroundColor": COLORS["card_bg"],
                "borderRadius": "10px",
                "padding": "15px",
                "marginRight": "10px"
            }),

            # Right: Performance by Opponent
            html.Div([
                html.H4("Performance by Opponent", style={"color": COLORS["text"]}),
                dcc.Graph(id="opponent-chart")
            ], style={
                "flex": "1",
                "backgroundColor": COLORS["card_bg"],
                "borderRadius": "10px",
                "padding": "15px"
            })
        ], style={"display": "flex", "marginBottom": "20px"}),

        # Row 4: Injury/News Status
        html.Div([
            html.H4("Latest News & Injury Status", style={"color": COLORS["text"]}),
            html.Div(id="injury-status-panel")
        ], style={
            "backgroundColor": COLORS["card_bg"],
            "borderRadius": "10px",
            "padding": "15px"
        })
    ], style={"padding": "20px"})


# =============================================================================
# APP LAYOUT
# =============================================================================

app.layout = html.Div([
    create_header(),
    create_player_selector(),
    create_main_content()
], style={
    "backgroundColor": COLORS["background"],
    "minHeight": "100vh",
    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
})


# =============================================================================
# CALLBACKS (Interactivity)
# =============================================================================

@callback(
    Output("stats-cards-container", "children"),
    Input("player-dropdown", "value")
)
def update_stats_cards(player_name):
    """
    Update the summary stats cards when a player is selected.

    WHAT IS A CALLBACK?
    A callback is a function that runs automatically when something changes.
    - Input: What triggers the callback (player dropdown value)
    - Output: What gets updated (stats cards)
    """
    if not player_name:
        return html.Div("Select a player to view stats")

    # Filter data for this player
    player_df = DF[DF["PLAYER_NAME"] == player_name]

    if player_df.empty:
        return html.Div("No data available for this player")

    # Calculate averages
    current_season = player_df[player_df["SEASON"] == "2024-25"]
    if current_season.empty:
        current_season = player_df

    avg_pts = current_season["PTS"].mean()
    avg_ast = current_season["AST"].mean()
    avg_reb = current_season["REB"].mean()
    games_played = len(current_season)

    # Create stat cards
    card_style = {
        "backgroundColor": COLORS["card_bg"],
        "borderRadius": "10px",
        "padding": "20px",
        "textAlign": "center",
        "flex": "1",
        "margin": "0 10px"
    }

    value_style = {
        "fontSize": "36px",
        "fontWeight": "bold",
        "color": COLORS["accent"]
    }

    label_style = {
        "color": "#888",
        "marginTop": "5px"
    }

    return html.Div([
        html.Div([
            html.Div(f"{avg_pts:.1f}", style=value_style),
            html.Div("PPG", style=label_style)
        ], style=card_style),
        html.Div([
            html.Div(f"{avg_ast:.1f}", style=value_style),
            html.Div("APG", style=label_style)
        ], style=card_style),
        html.Div([
            html.Div(f"{avg_reb:.1f}", style=value_style),
            html.Div("RPG", style=label_style)
        ], style=card_style),
        html.Div([
            html.Div(f"{games_played}", style={**value_style, "color": COLORS["secondary"]}),
            html.Div("Games", style=label_style)
        ], style=card_style),
    ], style={"display": "flex", "justifyContent": "center"})


@callback(
    Output("season-trend-chart", "figure"),
    Input("player-dropdown", "value")
)
def update_season_trend(player_name):
    """Create the season performance trend line chart."""
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name].copy()
    player_df = player_df.sort_values("_date")

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Points trend
    fig.add_trace(
        go.Scatter(
            x=player_df["_date"],
            y=player_df["PTS"],
            name="Points",
            line=dict(color=COLORS["accent"], width=2),
            mode="lines+markers",
            marker=dict(size=4)
        ),
        secondary_y=False
    )

    # Rolling average
    if "rolling_avg_pts_10" in player_df.columns:
        fig.add_trace(
            go.Scatter(
                x=player_df["_date"],
                y=player_df["rolling_avg_pts_10"],
                name="10-Game Avg",
                line=dict(color=COLORS["warning"], width=2, dash="dash"),
                mode="lines"
            ),
            secondary_y=False
        )

    # Style the chart
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    fig.update_yaxes(title_text="Points", secondary_y=False)

    return fig


@callback(
    Output("prediction-panel", "children"),
    Input("player-dropdown", "value")
)
def update_prediction(player_name):
    """Create the prediction panel with model predictions."""
    if not player_name:
        return html.Div("Select a player")

    predictions = {}

    for stat, predictor in PREDICTORS.items():
        try:
            result = predictor.predict_player_game(player_name, DF)
            if "error" not in result:
                pred_key = f"predicted_{stat.lower()}"
                predictions[stat] = {
                    "predicted": result.get(pred_key, "N/A"),
                    "recent_avg": result.get("recent_avg", "N/A"),
                    "confidence": result.get("confidence", "unknown")
                }
        except Exception as e:
            predictions[stat] = {"error": str(e)}

    if not predictions:
        return html.Div("No predictions available")

    # Confidence color mapping
    conf_colors = {
        "high": COLORS["accent"],
        "medium": COLORS["warning"],
        "low": COLORS["danger"]
    }

    pred_items = []
    for stat, data in predictions.items():
        if "error" in data:
            continue

        conf = data.get("confidence", "unknown")
        conf_color = conf_colors.get(conf, "#888")

        pred_items.append(html.Div([
            html.Div([
                html.Span(stat, style={"fontWeight": "bold", "marginRight": "10px"}),
                html.Span(
                    f"●",
                    style={"color": conf_color, "marginRight": "5px"}
                ),
                html.Span(conf.upper(), style={"fontSize": "10px", "color": conf_color})
            ]),
            html.Div([
                html.Span(
                    f"{data['predicted']}",
                    style={"fontSize": "28px", "fontWeight": "bold", "color": COLORS["accent"]}
                ),
                html.Span(
                    f" (avg: {data['recent_avg']})",
                    style={"color": "#888", "fontSize": "14px"}
                )
            ])
        ], style={
            "padding": "15px",
            "borderBottom": f"1px solid {COLORS['background']}"
        }))

    return html.Div(pred_items)


@callback(
    Output("home-away-chart", "figure"),
    Input("player-dropdown", "value")
)
def update_home_away_chart(player_name):
    """Create home vs away comparison bar chart."""
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name]

    home_stats = player_df[player_df["is_home"] == 1][["PTS", "AST", "REB"]].mean()
    away_stats = player_df[player_df["is_home"] == 0][["PTS", "AST", "REB"]].mean()

    fig = go.Figure()

    stats = ["PTS", "AST", "REB"]
    x = np.arange(len(stats))
    width = 0.35

    fig.add_trace(go.Bar(
        name="Home",
        x=stats,
        y=[home_stats.get(s, 0) for s in stats],
        marker_color=COLORS["accent"]
    ))

    fig.add_trace(go.Bar(
        name="Away",
        x=stats,
        y=[away_stats.get(s, 0) for s in stats],
        marker_color=COLORS["secondary"]
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=20, r=20, t=30, b=20),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


@callback(
    Output("opponent-chart", "figure"),
    Input("player-dropdown", "value")
)
def update_opponent_chart(player_name):
    """Create performance by opponent chart."""
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name]

    # Get average points by opponent
    if "opponent" in player_df.columns:
        opp_stats = player_df.groupby("opponent")["PTS"].agg(["mean", "count"])
        opp_stats = opp_stats[opp_stats["count"] >= 2]  # At least 2 games
        opp_stats = opp_stats.sort_values("mean", ascending=True).tail(10)

        fig = go.Figure(go.Bar(
            x=opp_stats["mean"],
            y=opp_stats.index,
            orientation="h",
            marker_color=COLORS["accent"],
            text=[f"{v:.1f}" for v in opp_stats["mean"]],
            textposition="outside"
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor=COLORS["card_bg"],
            margin=dict(l=50, r=50, t=30, b=20),
            xaxis_title="Avg Points",
            yaxis_title=""
        )

        return fig

    return go.Figure()


@callback(
    Output("injury-status-panel", "children"),
    Input("player-dropdown", "value")
)
def update_injury_status(player_name):
    """Fetch and display injury/news status."""
    if not player_name:
        return html.Div("Select a player")

    try:
        status = get_player_injury_status(player_name)

        status_colors = {
            "OUT": COLORS["danger"],
            "QUESTIONABLE": COLORS["warning"],
            "HEALTHY": COLORS["accent"],
            "UNKNOWN": "#888"
        }

        status_text = status.get("status", "UNKNOWN")
        status_color = status_colors.get(status_text, "#888")

        content = [
            html.Div([
                html.Span("Status: ", style={"color": "#888"}),
                html.Span(
                    status_text,
                    style={
                        "color": status_color,
                        "fontWeight": "bold",
                        "padding": "5px 10px",
                        "borderRadius": "5px",
                        "backgroundColor": f"{status_color}22"
                    }
                )
            ], style={"marginBottom": "15px"})
        ]

        # Show news if available
        news = status.get("news", [])
        if news:
            content.append(html.Div("Recent News:", style={"color": "#888", "marginBottom": "10px"}))
            for item in news[:3]:
                content.append(html.Div([
                    html.A(
                        item.get("title", ""),
                        href=item.get("link", "#"),
                        target="_blank",
                        style={"color": COLORS["secondary"]}
                    )
                ], style={"marginBottom": "5px", "fontSize": "14px"}))
        else:
            content.append(html.Div(
                "No recent news found",
                style={"color": "#666", "fontStyle": "italic"}
            ))

        return html.Div(content)

    except Exception as e:
        return html.Div(f"Unable to fetch news: {str(e)}", style={"color": "#888"})


# =============================================================================
# RUN THE APP
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🏀 NBA Player Stats Dashboard")
    print("=" * 50)
    print("\nOpen your browser to: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server\n")

    app.run(debug=True, port=8050)
