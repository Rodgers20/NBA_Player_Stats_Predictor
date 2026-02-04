# dashboard/app.py
"""
NBA Player Stats Dashboard
==========================
An interactive dashboard for analyzing NBA player performance
and making stat predictions.

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

app = Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/darkly/bootstrap.min.css"
    ],
    suppress_callback_exceptions=True
)

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
    "purple": "#9b59b6",  # Purple for combos
}

# Stat color mapping
STAT_COLORS = {
    "PTS": COLORS["accent"],
    "AST": COLORS["secondary"],
    "REB": COLORS["warning"],
    "PTS+AST": COLORS["purple"],
    "PTS+REB": "#e67e22",
    "PTS+AST+REB": "#1abc9c",
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
            style={"color": "#000"},
            searchable=True,
            clearable=False
        )
    ], style={
        "width": "400px",
        "margin": "0 auto",
        "padding": "20px"
    })


def create_main_content():
    """Create the main dashboard content area."""
    return html.Div([
        # Row 1: Stats cards
        html.Div(id="stats-cards-container", style={"marginBottom": "20px"}),

        # Row 2: Season Trend (Bar Chart) with stat selector
        html.Div([
            html.Div([
                html.Div([
                    html.H4("Game-by-Game Performance", style={"color": COLORS["text"], "display": "inline-block"}),
                    html.Div([
                        dcc.Dropdown(
                            id="trend-stat-dropdown",
                            options=[
                                {"label": "Points", "value": "PTS"},
                                {"label": "Assists", "value": "AST"},
                                {"label": "Rebounds", "value": "REB"},
                                {"label": "Points + Assists", "value": "PTS+AST"},
                                {"label": "Points + Rebounds", "value": "PTS+REB"},
                                {"label": "PTS + AST + REB", "value": "PTS+AST+REB"},
                            ],
                            value="PTS",
                            style={"width": "180px", "color": "#000"},
                            clearable=False
                        )
                    ], style={"display": "inline-block", "marginLeft": "20px", "verticalAlign": "middle"})
                ], style={"marginBottom": "10px"}),
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
                html.H4("Next Game Prediction", style={"color": COLORS["text"]}),
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

            # Right: Performance by Opponent with stat selector
            html.Div([
                html.Div([
                    html.H4("Performance by Opponent", style={"color": COLORS["text"], "display": "inline-block"}),
                    html.Div([
                        dcc.Dropdown(
                            id="opponent-stat-dropdown",
                            options=[
                                {"label": "Points", "value": "PTS"},
                                {"label": "Assists", "value": "AST"},
                                {"label": "Rebounds", "value": "REB"},
                                {"label": "PTS + AST", "value": "PTS+AST"},
                                {"label": "PTS + REB", "value": "PTS+REB"},
                                {"label": "PTS + AST + REB", "value": "PTS+AST+REB"},
                            ],
                            value="PTS",
                            style={"width": "150px", "color": "#000"},
                            clearable=False
                        )
                    ], style={"display": "inline-block", "marginLeft": "15px", "verticalAlign": "middle"})
                ], style={"marginBottom": "10px"}),
                dcc.Graph(id="opponent-chart")
            ], style={
                "flex": "1",
                "backgroundColor": COLORS["card_bg"],
                "borderRadius": "10px",
                "padding": "15px"
            })
        ], style={"display": "flex", "marginBottom": "20px"}),

        # Row 4: Player Status
        html.Div([
            html.H4("Player Status", style={"color": COLORS["text"]}),
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
# HELPER FUNCTIONS
# =============================================================================

def calculate_stat(df, stat_name):
    """Calculate a stat value, supporting combo stats like PTS+AST."""
    if "+" in stat_name:
        parts = stat_name.split("+")
        return sum(df[p] for p in parts)
    return df[stat_name]


def get_stat_label(stat_name):
    """Get human-readable label for a stat."""
    labels = {
        "PTS": "Points",
        "AST": "Assists",
        "REB": "Rebounds",
        "PTS+AST": "Points + Assists",
        "PTS+REB": "Points + Rebounds",
        "PTS+AST+REB": "PTS + AST + REB",
    }
    return labels.get(stat_name, stat_name)


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output("stats-cards-container", "children"),
    Input("player-dropdown", "value")
)
def update_stats_cards(player_name):
    """Update the summary stats cards when a player is selected."""
    if not player_name:
        return html.Div("Select a player to view stats")

    player_df = DF[DF["PLAYER_NAME"] == player_name]

    if player_df.empty:
        return html.Div("No data available for this player")

    # Get current season or all data
    current_season = player_df[player_df["SEASON"] == "2024-25"]
    if current_season.empty:
        current_season = player_df

    avg_pts = current_season["PTS"].mean()
    avg_ast = current_season["AST"].mean()
    avg_reb = current_season["REB"].mean()
    games_played = len(current_season)

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
    }

    label_style = {
        "color": "#888",
        "marginTop": "5px"
    }

    return html.Div([
        html.Div([
            html.Div(f"{avg_pts:.1f}", style={**value_style, "color": STAT_COLORS["PTS"]}),
            html.Div("PPG", style=label_style)
        ], style=card_style),
        html.Div([
            html.Div(f"{avg_ast:.1f}", style={**value_style, "color": STAT_COLORS["AST"]}),
            html.Div("APG", style=label_style)
        ], style=card_style),
        html.Div([
            html.Div(f"{avg_reb:.1f}", style={**value_style, "color": STAT_COLORS["REB"]}),
            html.Div("RPG", style=label_style)
        ], style=card_style),
        html.Div([
            html.Div(f"{games_played}", style={**value_style, "color": "#888"}),
            html.Div("Games", style=label_style)
        ], style=card_style),
    ], style={"display": "flex", "justifyContent": "center"})


@callback(
    Output("season-trend-chart", "figure"),
    [Input("player-dropdown", "value"),
     Input("trend-stat-dropdown", "value")]
)
def update_season_trend(player_name, stat_name):
    """Create a bar chart showing game-by-game performance."""
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name].copy()
    player_df = player_df.sort_values("_date")

    # Get last 25 games for cleaner visualization
    player_df = player_df.tail(25)

    # Calculate the stat value
    stat_values = calculate_stat(player_df, stat_name)
    avg_value = stat_values.mean()

    # Get color for this stat
    bar_color = STAT_COLORS.get(stat_name, COLORS["accent"])

    # Create bar chart
    fig = go.Figure()

    # Add bars for each game
    fig.add_trace(go.Bar(
        x=player_df["_date"],
        y=stat_values,
        marker_color=bar_color,
        name=get_stat_label(stat_name),
        text=[f"{v:.0f}" for v in stat_values],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate=(
            "<b>%{x|%b %d}</b><br>" +
            f"{get_stat_label(stat_name)}: " + "%{y:.0f}<br>" +
            "<extra></extra>"
        )
    ))

    # Add average line
    fig.add_hline(
        y=avg_value,
        line_dash="dash",
        line_color="#ffffff",
        line_width=2,
        annotation_text=f"Avg: {avg_value:.1f}",
        annotation_position="right",
        annotation_font_color="#ffffff"
    )

    # Style the chart
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
        xaxis_title="",
        yaxis_title=get_stat_label(stat_name),
        bargap=0.3,
    )

    fig.update_xaxes(
        tickformat="%b %d",
        tickangle=-45
    )

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
        stat_color = STAT_COLORS.get(stat, COLORS["accent"])

        pred_items.append(html.Div([
            html.Div([
                html.Span(stat, style={"fontWeight": "bold", "marginRight": "10px", "color": stat_color}),
                html.Span("●", style={"color": conf_color, "marginRight": "5px"}),
                html.Span(conf.upper(), style={"fontSize": "10px", "color": conf_color})
            ]),
            html.Div([
                html.Span(
                    f"{data['predicted']}",
                    style={"fontSize": "32px", "fontWeight": "bold", "color": stat_color}
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

    home_df = player_df[player_df["is_home"] == 1]
    away_df = player_df[player_df["is_home"] == 0]

    home_stats = {
        "PTS": home_df["PTS"].mean() if len(home_df) > 0 else 0,
        "AST": home_df["AST"].mean() if len(home_df) > 0 else 0,
        "REB": home_df["REB"].mean() if len(home_df) > 0 else 0,
    }

    away_stats = {
        "PTS": away_df["PTS"].mean() if len(away_df) > 0 else 0,
        "AST": away_df["AST"].mean() if len(away_df) > 0 else 0,
        "REB": away_df["REB"].mean() if len(away_df) > 0 else 0,
    }

    fig = go.Figure()

    stats = ["PTS", "AST", "REB"]

    fig.add_trace(go.Bar(
        name=f"Home ({len(home_df)} games)",
        x=stats,
        y=[home_stats[s] for s in stats],
        marker_color=COLORS["accent"],
        text=[f"{home_stats[s]:.1f}" for s in stats],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name=f"Away ({len(away_df)} games)",
        x=stats,
        y=[away_stats[s] for s in stats],
        marker_color=COLORS["secondary"],
        text=[f"{away_stats[s]:.1f}" for s in stats],
        textposition="outside"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=20, r=20, t=40, b=20),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig


@callback(
    Output("opponent-chart", "figure"),
    [Input("player-dropdown", "value"),
     Input("opponent-stat-dropdown", "value")]
)
def update_opponent_chart(player_name, stat_name):
    """Create performance by opponent chart with selectable stat."""
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name]

    if "opponent" not in player_df.columns:
        return go.Figure()

    # Calculate the stat for each game
    player_df = player_df.copy()
    player_df["_stat_value"] = calculate_stat(player_df, stat_name)

    # Group by opponent
    opp_stats = player_df.groupby("opponent").agg(
        avg=("_stat_value", "mean"),
        games=("_stat_value", "count")
    ).reset_index()

    # Filter to opponents with at least 1 game, show top 10
    opp_stats = opp_stats[opp_stats["games"] >= 1]
    opp_stats = opp_stats.sort_values("avg", ascending=True).tail(10)

    bar_color = STAT_COLORS.get(stat_name, COLORS["accent"])

    fig = go.Figure(go.Bar(
        x=opp_stats["avg"],
        y=opp_stats["opponent"],
        orientation="h",
        marker_color=bar_color,
        text=[f"{v:.1f} ({g}g)" for v, g in zip(opp_stats["avg"], opp_stats["games"])],
        textposition="outside",
        hovertemplate=(
            "<b>vs %{y}</b><br>" +
            f"Avg {get_stat_label(stat_name)}: " + "%{x:.1f}<br>" +
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=60, r=80, t=20, b=20),
        xaxis_title=f"Avg {get_stat_label(stat_name)}",
        yaxis_title=""
    )

    return fig


@callback(
    Output("injury-status-panel", "children"),
    Input("player-dropdown", "value")
)
def update_injury_status(player_name):
    """Display player status - ACTIVE by default unless injury news found."""
    if not player_name:
        return html.Div("Select a player")

    try:
        status = get_player_injury_status(player_name)

        # Map status to display text
        status_text = status.get("status", "UNKNOWN")

        # If UNKNOWN or HEALTHY, show as ACTIVE (more user-friendly)
        if status_text in ["UNKNOWN", "HEALTHY"]:
            status_text = "ACTIVE"

        # Status colors
        status_colors = {
            "ACTIVE": COLORS["accent"],
            "OUT": COLORS["danger"],
            "QUESTIONABLE": COLORS["warning"],
            "DOUBTFUL": COLORS["warning"],
        }

        status_color = status_colors.get(status_text, COLORS["accent"])

        # Get game count from our data
        player_df = DF[DF["PLAYER_NAME"] == player_name]
        current_season = player_df[player_df["SEASON"] == "2024-25"]
        games_this_season = len(current_season) if not current_season.empty else len(player_df)

        # Build the content
        content = [
            # Status badge
            html.Div([
                html.Span(
                    status_text,
                    style={
                        "color": "#fff",
                        "fontWeight": "bold",
                        "padding": "8px 20px",
                        "borderRadius": "20px",
                        "backgroundColor": status_color,
                        "fontSize": "16px"
                    }
                )
            ], style={"marginBottom": "20px"}),

            # Quick stats row
            html.Div([
                html.Div([
                    html.Span(f"{games_this_season}", style={"fontSize": "24px", "fontWeight": "bold", "color": COLORS["text"]}),
                    html.Span(" games played this season", style={"color": "#888", "marginLeft": "5px"})
                ]),
            ], style={"marginBottom": "15px"}),
        ]

        # Show news only if we found relevant articles
        news = status.get("news", [])
        if news:
            content.append(html.Div([
                html.Div("Recent News:", style={"color": "#888", "marginBottom": "10px", "fontWeight": "bold"}),
            ]))
            for item in news[:3]:
                title = item.get("title", "").strip()
                if title:
                    content.append(html.Div([
                        html.A(
                            title[:100] + "..." if len(title) > 100 else title,
                            href=item.get("link", "#"),
                            target="_blank",
                            style={"color": COLORS["secondary"], "textDecoration": "none"}
                        )
                    ], style={"marginBottom": "8px", "fontSize": "14px", "paddingLeft": "10px"}))
        else:
            content.append(html.Div(
                "No injury reports - player appears healthy",
                style={"color": COLORS["accent"], "fontStyle": "italic"}
            ))

        return html.Div(content)

    except Exception as e:
        # Default to ACTIVE if we can't fetch news
        return html.Div([
            html.Span(
                "ACTIVE",
                style={
                    "color": "#fff",
                    "fontWeight": "bold",
                    "padding": "8px 20px",
                    "borderRadius": "20px",
                    "backgroundColor": COLORS["accent"],
                    "fontSize": "16px"
                }
            ),
            html.P("No injury reports found", style={"color": "#888", "marginTop": "15px"})
        ])


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
