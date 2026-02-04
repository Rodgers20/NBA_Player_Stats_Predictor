# dashboard/app.py
"""
NBA Player Stats Dashboard - Outlier.bet Style
===============================================
A sleek, dark-themed dashboard inspired by outlier.bet's design.

TO RUN:
    python dashboard/app.py
    Then open http://127.0.0.1:8050 in your browser
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
    return df


def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    predictors = {}
    for target in ["pts", "ast", "reb"]:
        filepath = os.path.join(models_dir, f"{target}_predictor.pkl")
        if os.path.exists(filepath):
            predictors[target.upper()] = NBAPredictor.load(filepath)
    return predictors


print("Loading data...")
DF = load_data()
PREDICTORS = load_models()
PLAYERS = sorted(DF["PLAYER_NAME"].unique().tolist())
print(f"Loaded {len(DF)} game records for {len(PLAYERS)} players")

# =============================================================================
# OUTLIER-STYLE COLOR SCHEME
# =============================================================================

COLORS = {
    # Backgrounds
    "bg_primary": "#050505",      # Near black (main background)
    "bg_card": "#0d0d0d",         # Slightly lighter for cards
    "bg_hover": "#1a1a1a",        # Hover state
    "border": "#1b1b1b",          # Subtle borders

    # Text
    "text_primary": "#ffffff",
    "text_secondary": "#888888",
    "text_muted": "#555555",

    # Accent (Outlier purple)
    "accent": "#9d8ec9",          # Muted purple
    "accent_bright": "#b8a9e0",   # Brighter purple

    # Traffic Light Hit Rates
    "hit_100": "#22c55e",         # Light green (80-100%)
    "hit_80": "#16a34a",          # Green (66-80%)
    "hit_65": "#15803d",          # Dark green (51-65%)
    "hit_50": "#404040",          # Neutral (50%)
    "hit_35": "#991b1b",          # Dark red (35-49%)
    "hit_20": "#dc2626",          # Red (15-34%)
    "hit_0": "#ef4444",           # Light red (0-14%)

    # Status
    "active": "#22c55e",
    "questionable": "#eab308",
    "out": "#ef4444",
}


def get_hit_rate_color(hit_rate):
    """Get traffic light color based on hit rate percentage."""
    if hit_rate >= 80:
        return COLORS["hit_100"]
    elif hit_rate >= 66:
        return COLORS["hit_80"]
    elif hit_rate >= 51:
        return COLORS["hit_65"]
    elif hit_rate >= 50:
        return COLORS["hit_50"]
    elif hit_rate >= 35:
        return COLORS["hit_35"]
    elif hit_rate >= 15:
        return COLORS["hit_20"]
    else:
        return COLORS["hit_0"]


# =============================================================================
# DASH APP SETUP
# =============================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# =============================================================================
# STYLES
# =============================================================================

CARD_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "16px",
}

STAT_BOX_STYLE = {
    "backgroundColor": COLORS["bg_primary"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "6px",
    "padding": "16px",
    "textAlign": "center",
    "minWidth": "100px",
}

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("NBA Props", style={
                "color": COLORS["text_primary"],
                "fontSize": "24px",
                "fontWeight": "600",
                "margin": "0",
                "display": "inline-block"
            }),
            html.Span(" BETA", style={
                "color": COLORS["accent"],
                "fontSize": "12px",
                "marginLeft": "8px",
                "padding": "2px 8px",
                "backgroundColor": f"{COLORS['accent']}22",
                "borderRadius": "4px",
            })
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "padding": "16px 24px",
        "borderBottom": f"1px solid {COLORS['border']}",
    }),

    # Main Content
    html.Div([
        # Sidebar - Player Selection
        html.Div([
            html.Div([
                html.Label("PLAYER", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "letterSpacing": "1px",
                    "marginBottom": "8px",
                    "display": "block"
                }),
                dcc.Dropdown(
                    id="player-dropdown",
                    options=[{"label": p, "value": p} for p in PLAYERS],
                    value=PLAYERS[0] if PLAYERS else None,
                    placeholder="Search player...",
                    style={"backgroundColor": COLORS["bg_card"]}
                )
            ], style=CARD_STYLE),

            # Player Status Card
            html.Div(id="status-card", style=CARD_STYLE),

            # Prediction Card
            html.Div(id="prediction-card", style=CARD_STYLE),

        ], style={"width": "320px", "flexShrink": "0"}),

        # Main Panel
        html.Div([
            # Season Stats Header
            html.Div(id="season-stats-header"),

            # Hit Rate Tables
            html.Div(id="hit-rate-section"),

            # Game Log
            html.Div(id="game-log-section"),

            # Splits Analysis
            html.Div(id="splits-section"),

        ], style={"flex": "1", "marginLeft": "24px"}),

    ], style={
        "display": "flex",
        "padding": "24px",
        "maxWidth": "1400px",
        "margin": "0 auto",
    }),

], style={
    "backgroundColor": COLORS["bg_primary"],
    "minHeight": "100vh",
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "color": COLORS["text_primary"],
})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output("status-card", "children"),
    Input("player-dropdown", "value")
)
def update_status_card(player_name):
    if not player_name:
        return None

    try:
        status = get_player_injury_status(player_name)
        status_text = status.get("status", "UNKNOWN")
        if status_text in ["UNKNOWN", "HEALTHY"]:
            status_text = "ACTIVE"
    except:
        status_text = "ACTIVE"

    status_colors = {
        "ACTIVE": COLORS["active"],
        "OUT": COLORS["out"],
        "QUESTIONABLE": COLORS["questionable"],
    }
    status_color = status_colors.get(status_text, COLORS["active"])

    player_df = DF[DF["PLAYER_NAME"] == player_name]
    games = len(player_df[player_df["SEASON"] == "2024-25"])
    if games == 0:
        games = len(player_df)

    return html.Div([
        html.Div([
            html.Label("STATUS", style={
                "color": COLORS["text_secondary"],
                "fontSize": "11px",
                "fontWeight": "600",
                "letterSpacing": "1px",
            }),
        ]),
        html.Div([
            html.Span("●", style={"color": status_color, "marginRight": "8px"}),
            html.Span(status_text, style={
                "color": status_color,
                "fontWeight": "600",
                "fontSize": "14px"
            })
        ], style={"marginTop": "8px"}),
        html.Div(f"{games} games played", style={
            "color": COLORS["text_muted"],
            "fontSize": "13px",
            "marginTop": "4px"
        })
    ])


@callback(
    Output("prediction-card", "children"),
    Input("player-dropdown", "value")
)
def update_prediction_card(player_name):
    if not player_name:
        return None

    predictions = []
    for stat, predictor in PREDICTORS.items():
        try:
            result = predictor.predict_player_game(player_name, DF)
            if "error" not in result:
                pred_key = f"predicted_{stat.lower()}"
                predictions.append({
                    "stat": stat,
                    "predicted": result.get(pred_key, 0),
                    "recent_avg": result.get("recent_avg", 0),
                    "confidence": result.get("confidence", "low")
                })
        except:
            pass

    if not predictions:
        return html.Div("No predictions available")

    conf_colors = {"high": COLORS["active"], "medium": COLORS["questionable"], "low": COLORS["text_muted"]}

    return html.Div([
        html.Label("NEXT GAME PROJECTION", style={
            "color": COLORS["text_secondary"],
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "marginBottom": "16px",
            "display": "block"
        }),
        html.Div([
            html.Div([
                html.Div(pred["stat"], style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "12px",
                    "marginBottom": "4px"
                }),
                html.Div([
                    html.Span(f"{pred['predicted']}", style={
                        "color": COLORS["accent_bright"],
                        "fontSize": "28px",
                        "fontWeight": "600"
                    }),
                ]),
                html.Div([
                    html.Span("●", style={
                        "color": conf_colors.get(pred["confidence"], COLORS["text_muted"]),
                        "fontSize": "8px",
                        "marginRight": "4px"
                    }),
                    html.Span(f"avg {pred['recent_avg']}", style={
                        "color": COLORS["text_muted"],
                        "fontSize": "12px"
                    })
                ])
            ], style={
                **STAT_BOX_STYLE,
                "flex": "1",
                "margin": "0 4px"
            }) for pred in predictions
        ], style={"display": "flex"})
    ])


@callback(
    Output("season-stats-header", "children"),
    Input("player-dropdown", "value")
)
def update_season_stats(player_name):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name]
    current = player_df[player_df["SEASON"] == "2024-25"]
    if current.empty:
        current = player_df

    stats = {
        "PTS": current["PTS"].mean(),
        "AST": current["AST"].mean(),
        "REB": current["REB"].mean(),
        "FG%": current["FG_PCT"].mean() * 100 if "FG_PCT" in current.columns else 0,
        "3P%": current["FG3_PCT"].mean() * 100 if "FG3_PCT" in current.columns else 0,
    }

    return html.Div([
        html.Div([
            html.H2(player_name, style={
                "color": COLORS["text_primary"],
                "fontSize": "20px",
                "fontWeight": "600",
                "margin": "0"
            }),
            html.Span("2024-25 Season", style={
                "color": COLORS["text_muted"],
                "fontSize": "13px",
                "marginLeft": "12px"
            })
        ], style={"marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Div(f"{val:.1f}", style={
                    "color": COLORS["text_primary"],
                    "fontSize": "24px",
                    "fontWeight": "600"
                }),
                html.Div(key, style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "12px"
                })
            ], style={
                **STAT_BOX_STYLE,
                "marginRight": "8px"
            }) for key, val in stats.items()
        ], style={"display": "flex", "flexWrap": "wrap"})
    ], style={**CARD_STYLE, "marginBottom": "24px"})


@callback(
    Output("hit-rate-section", "children"),
    Input("player-dropdown", "value")
)
def update_hit_rates(player_name):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    def calc_hit_rate(df, stat, threshold, n_games):
        recent = df.head(n_games)
        if len(recent) == 0:
            return 0
        hits = (recent[stat] >= threshold).sum()
        return (hits / len(recent)) * 100

    # Common betting lines (approximate)
    props = [
        {"stat": "PTS", "label": "Points", "lines": [15.5, 20.5, 25.5, 30.5]},
        {"stat": "AST", "label": "Assists", "lines": [4.5, 6.5, 8.5, 10.5]},
        {"stat": "REB", "label": "Rebounds", "lines": [4.5, 6.5, 8.5, 10.5]},
    ]

    season_avg = {
        "PTS": player_df["PTS"].mean(),
        "AST": player_df["AST"].mean(),
        "REB": player_df["REB"].mean(),
    }

    tables = []
    for prop in props:
        stat = prop["stat"]
        avg = season_avg[stat]

        # Find closest line to average
        closest_line = min(prop["lines"], key=lambda x: abs(x - avg))

        rows = []
        for line in prop["lines"]:
            l5 = calc_hit_rate(player_df, stat, line, 5)
            l10 = calc_hit_rate(player_df, stat, line, 10)
            l20 = calc_hit_rate(player_df, stat, line, 20)
            season = calc_hit_rate(player_df, stat, line, len(player_df))

            is_closest = line == closest_line

            rows.append(html.Tr([
                html.Td(f"Over {line}", style={
                    "padding": "10px 12px",
                    "color": COLORS["accent"] if is_closest else COLORS["text_primary"],
                    "fontWeight": "600" if is_closest else "400",
                    "borderBottom": f"1px solid {COLORS['border']}"
                }),
                html.Td([
                    html.Span(f"{l5:.0f}%", style={"color": get_hit_rate_color(l5)})
                ], style={"padding": "10px 12px", "textAlign": "center", "borderBottom": f"1px solid {COLORS['border']}"}),
                html.Td([
                    html.Span(f"{l10:.0f}%", style={"color": get_hit_rate_color(l10)})
                ], style={"padding": "10px 12px", "textAlign": "center", "borderBottom": f"1px solid {COLORS['border']}"}),
                html.Td([
                    html.Span(f"{l20:.0f}%", style={"color": get_hit_rate_color(l20)})
                ], style={"padding": "10px 12px", "textAlign": "center", "borderBottom": f"1px solid {COLORS['border']}"}),
                html.Td([
                    html.Span(f"{season:.0f}%", style={"color": get_hit_rate_color(season)})
                ], style={"padding": "10px 12px", "textAlign": "center", "borderBottom": f"1px solid {COLORS['border']}"}),
            ]))

        tables.append(html.Div([
            html.Div([
                html.Span(prop["label"], style={
                    "color": COLORS["text_primary"],
                    "fontWeight": "600",
                    "fontSize": "14px"
                }),
                html.Span(f" AVG {avg:.1f}", style={
                    "color": COLORS["accent"],
                    "fontSize": "13px",
                    "marginLeft": "8px"
                })
            ], style={"marginBottom": "12px"}),

            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Line", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("L5", style={"padding": "8px 12px", "textAlign": "center", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("L10", style={"padding": "8px 12px", "textAlign": "center", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("L20", style={"padding": "8px 12px", "textAlign": "center", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("Season", style={"padding": "8px 12px", "textAlign": "center", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                    ])
                ]),
                html.Tbody(rows)
            ], style={"width": "100%", "borderCollapse": "collapse"})
        ], style={**CARD_STYLE, "flex": "1", "marginRight": "16px" if prop != props[-1] else "0"}))

    return html.Div([
        html.Label("HIT RATES", style={
            "color": COLORS["text_secondary"],
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "marginBottom": "16px",
            "display": "block"
        }),
        html.Div(tables, style={"display": "flex"})
    ])


@callback(
    Output("game-log-section", "children"),
    Input("player-dropdown", "value")
)
def update_game_log(player_name):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False).head(10)

    rows = []
    for _, game in player_df.iterrows():
        date_str = game["_date"].strftime("%b %d") if pd.notna(game["_date"]) else "N/A"
        matchup = game.get("MATCHUP", "N/A")
        opp = matchup.split("@")[-1].split("vs.")[-1].strip()[:3] if isinstance(matchup, str) else "N/A"
        wl = game.get("WL", "-")
        pts = int(game.get("PTS", 0))
        ast = int(game.get("AST", 0))
        reb = int(game.get("REB", 0))
        mins = int(game.get("MIN", 0)) if pd.notna(game.get("MIN")) else 0

        wl_color = COLORS["active"] if wl == "W" else COLORS["out"] if wl == "L" else COLORS["text_muted"]

        rows.append(html.Tr([
            html.Td(date_str, style={"padding": "10px 12px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(opp, style={"padding": "10px 12px", "fontWeight": "500", "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(wl, style={"padding": "10px 12px", "color": wl_color, "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(str(mins), style={"padding": "10px 12px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(str(pts), style={"padding": "10px 12px", "fontWeight": "600", "color": COLORS["accent_bright"], "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(str(ast), style={"padding": "10px 12px", "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(str(reb), style={"padding": "10px 12px", "borderBottom": f"1px solid {COLORS['border']}"}),
        ]))

    return html.Div([
        html.Label("GAME LOG", style={
            "color": COLORS["text_secondary"],
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "marginBottom": "16px",
            "display": "block"
        }),
        html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Date", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("Opp", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("W/L", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("MIN", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("PTS", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("AST", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("REB", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                    ])
                ]),
                html.Tbody(rows)
            ], style={"width": "100%", "borderCollapse": "collapse"})
        ], style=CARD_STYLE)
    ], style={"marginTop": "24px"})


@callback(
    Output("splits-section", "children"),
    Input("player-dropdown", "value")
)
def update_splits(player_name):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name]

    # Home vs Away
    home = player_df[player_df["is_home"] == 1]
    away = player_df[player_df["is_home"] == 0]

    splits_data = [
        {
            "label": "Home",
            "games": len(home),
            "pts": home["PTS"].mean() if len(home) > 0 else 0,
            "ast": home["AST"].mean() if len(home) > 0 else 0,
            "reb": home["REB"].mean() if len(home) > 0 else 0,
        },
        {
            "label": "Away",
            "games": len(away),
            "pts": away["PTS"].mean() if len(away) > 0 else 0,
            "ast": away["AST"].mean() if len(away) > 0 else 0,
            "reb": away["REB"].mean() if len(away) > 0 else 0,
        },
    ]

    rows = []
    for split in splits_data:
        rows.append(html.Tr([
            html.Td(split["label"], style={"padding": "10px 12px", "fontWeight": "500", "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(str(split["games"]), style={"padding": "10px 12px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(f"{split['pts']:.1f}", style={"padding": "10px 12px", "fontWeight": "600", "color": COLORS["accent_bright"], "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(f"{split['ast']:.1f}", style={"padding": "10px 12px", "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(f"{split['reb']:.1f}", style={"padding": "10px 12px", "borderBottom": f"1px solid {COLORS['border']}"}),
        ]))

    return html.Div([
        html.Label("SPLITS", style={
            "color": COLORS["text_secondary"],
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "marginBottom": "16px",
            "display": "block"
        }),
        html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Split", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("GP", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("PTS", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("AST", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                        html.Th("REB", style={"padding": "8px 12px", "textAlign": "left", "color": COLORS["text_secondary"], "fontSize": "11px", "fontWeight": "600", "borderBottom": f"1px solid {COLORS['border']}"}),
                    ])
                ]),
                html.Tbody(rows)
            ], style={"width": "100%", "borderCollapse": "collapse"})
        ], style=CARD_STYLE)
    ], style={"marginTop": "24px"})


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🏀 NBA Props Dashboard")
    print("=" * 50)
    print("\nOpen your browser to: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server\n")

    app.run(debug=True, port=8050)
