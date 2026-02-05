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
    "ast": "#f97066",       # Coral/salmon for assists
    "reb": "#a78bfa",       # Purple for rebounds

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
    "padding": "8px 16px",
    "borderRadius": "6px",
    "cursor": "pointer",
    "fontSize": "13px",
    "fontWeight": "500",
    "color": COLORS["text_secondary"],
    "backgroundColor": "transparent",
    "border": "none",
    "marginRight": "4px",
}

TAB_ACTIVE = {
    **TAB_STYLE,
    "backgroundColor": COLORS["border"],
    "color": COLORS["text"],
}

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div([
    # Header with player info
    html.Div([
        html.Div([
            dcc.Dropdown(
                id="player-dropdown",
                options=[{"label": p, "value": p} for p in PLAYERS],
                value=PLAYERS[0] if PLAYERS else None,
                placeholder="Search player...",
                style={"width": "300px", "backgroundColor": COLORS["card"]}
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "20px"}),

        html.Div(id="player-header", style={"marginTop": "16px"}),
    ], style={
        "padding": "20px 24px",
        "backgroundColor": COLORS["card"],
        "borderBottom": f"1px solid {COLORS['border']}"
    }),

    # Main content
    html.Div([
        # Left panel - Main chart area
        html.Div([
            # Stat type tabs
            html.Div([
                html.Button("PTS", id="tab-pts", n_clicks=0, style=TAB_ACTIVE),
                html.Button("AST", id="tab-ast", n_clicks=0, style=TAB_STYLE),
                html.Button("REB", id="tab-reb", n_clicks=0, style=TAB_STYLE),
                html.Button("PTS+AST", id="tab-pts-ast", n_clicks=0, style=TAB_STYLE),
                html.Button("PTS+REB", id="tab-pts-reb", n_clicks=0, style=TAB_STYLE),
                html.Button("PTS+AST+REB", id="tab-all", n_clicks=0, style=TAB_STYLE),
            ], style={"display": "flex", "marginBottom": "16px", "flexWrap": "wrap"}),

            # Store for selected stat
            dcc.Store(id="selected-stat", data="PTS"),

            # Time period tabs
            html.Div([
                html.Button("L5", id="period-l5", n_clicks=0, style=TAB_STYLE),
                html.Button("L10", id="period-l10", n_clicks=0, style=TAB_ACTIVE),
                html.Button("L20", id="period-l20", n_clicks=0, style=TAB_STYLE),
                html.Button("Season", id="period-season", n_clicks=0, style=TAB_STYLE),
            ], style={"display": "flex", "marginBottom": "20px"}),

            dcc.Store(id="selected-period", data=10),

            # Hit rate header
            html.Div(id="hit-rate-header", style={"marginBottom": "20px"}),

            # Main chart
            html.Div([
                dcc.Graph(id="main-chart", config={"displayModeBar": False})
            ], style=CARD),

            # Supporting stats
            html.Div(id="supporting-stats", style=CARD),

        ], style={"flex": "2", "marginRight": "20px"}),

        # Right panel - Sidebar
        html.Div([
            # Prediction card
            html.Div(id="prediction-card", style=CARD),

            # Status card
            html.Div(id="status-card", style=CARD),

            # Hit rates table
            html.Div(id="hit-rates-table", style=CARD),

            # Splits
            html.Div(id="splits-card", style=CARD),

        ], style={"width": "350px", "flexShrink": "0"}),

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
    [Output("selected-stat", "data"),
     Output("tab-pts", "style"),
     Output("tab-ast", "style"),
     Output("tab-reb", "style"),
     Output("tab-pts-ast", "style"),
     Output("tab-pts-reb", "style"),
     Output("tab-all", "style")],
    [Input("tab-pts", "n_clicks"),
     Input("tab-ast", "n_clicks"),
     Input("tab-reb", "n_clicks"),
     Input("tab-pts-ast", "n_clicks"),
     Input("tab-pts-reb", "n_clicks"),
     Input("tab-all", "n_clicks")]
)
def update_stat_tabs(pts, ast, reb, pts_ast, pts_reb, all_stats):
    from dash import ctx
    triggered = ctx.triggered_id

    styles = [TAB_STYLE] * 6
    stat = "PTS"

    if triggered == "tab-pts" or triggered is None:
        stat, styles[0] = "PTS", TAB_ACTIVE
    elif triggered == "tab-ast":
        stat, styles[1] = "AST", TAB_ACTIVE
    elif triggered == "tab-reb":
        stat, styles[2] = "REB", TAB_ACTIVE
    elif triggered == "tab-pts-ast":
        stat, styles[3] = "PTS+AST", TAB_ACTIVE
    elif triggered == "tab-pts-reb":
        stat, styles[4] = "PTS+REB", TAB_ACTIVE
    elif triggered == "tab-all":
        stat, styles[5] = "PTS+AST+REB", TAB_ACTIVE

    return [stat] + styles


# Tab click handlers for period
@callback(
    [Output("selected-period", "data"),
     Output("period-l5", "style"),
     Output("period-l10", "style"),
     Output("period-l20", "style"),
     Output("period-season", "style")],
    [Input("period-l5", "n_clicks"),
     Input("period-l10", "n_clicks"),
     Input("period-l20", "n_clicks"),
     Input("period-season", "n_clicks")]
)
def update_period_tabs(l5, l10, l20, season):
    from dash import ctx
    triggered = ctx.triggered_id

    styles = [TAB_STYLE] * 4
    period = 10

    if triggered == "period-l5":
        period, styles[0] = 5, TAB_ACTIVE
    elif triggered == "period-l10" or triggered is None:
        period, styles[1] = 10, TAB_ACTIVE
    elif triggered == "period-l20":
        period, styles[2] = 20, TAB_ACTIVE
    elif triggered == "period-season":
        period, styles[3] = 100, TAB_ACTIVE

    return [period] + styles


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

    avg_pts = current["PTS"].mean()
    avg_ast = current["AST"].mean()
    avg_reb = current["REB"].mean()

    return html.Div([
        html.H1(player_name, style={
            "fontSize": "24px",
            "fontWeight": "600",
            "margin": "0 0 8px 0"
        }),
        html.Div([
            html.Span(f"{avg_pts:.1f} PTS", style={"color": COLORS["pts"], "marginRight": "16px", "fontWeight": "500"}),
            html.Span(f"{avg_ast:.1f} AST", style={"color": COLORS["ast"], "marginRight": "16px", "fontWeight": "500"}),
            html.Span(f"{avg_reb:.1f} REB", style={"color": COLORS["reb"], "fontWeight": "500"}),
        ])
    ])


@callback(
    Output("hit-rate-header", "children"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data"),
     Input("selected-period", "data")]
)
def update_hit_rate_header(player_name, stat, period):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    # Calculate stat values
    if "+" in stat:
        parts = stat.split("+")
        player_df["_stat"] = sum(player_df[p] for p in parts)
    else:
        player_df["_stat"] = player_df[stat]

    avg = player_df["_stat"].head(period).mean()
    threshold = round(avg - 0.5) + 0.5  # Round to .5

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
    season_pct, _ = calc_hit(player_df, len(player_df))

    return html.Div([
        html.Div([
            html.Span("% ", style={"color": COLORS["accent"], "fontSize": "18px"}),
            html.Span(f"{player_name} - {stat} ", style={"fontSize": "16px", "fontWeight": "500"}),
        ], style={"marginBottom": "12px"}),

        html.Div([
            # Main hit rate
            html.Div([
                html.Div(f"Last {period}", style={"color": COLORS["text_secondary"], "fontSize": "12px"}),
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
            ], style={"textAlign": "center", "marginRight": "20px"}),

            html.Div([
                html.Div("L20", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{l20_pct:.0f}%", style={"color": get_hit_color(l20_pct), "fontWeight": "600"})
            ], style={"textAlign": "center", "marginRight": "20px"}),

            html.Div([
                html.Div("Season", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{season_pct:.0f}%", style={"color": get_hit_color(season_pct), "fontWeight": "600"})
            ], style={"textAlign": "center"}),

        ], style={"display": "flex", "alignItems": "flex-end"}),
    ])


@callback(
    Output("main-chart", "figure"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data"),
     Input("selected-period", "data")]
)
def update_main_chart(player_name, stat, period):
    if not player_name:
        return go.Figure()

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False).head(period)
    player_df = player_df.iloc[::-1]  # Reverse for chronological order

    fig = go.Figure()

    # Determine if stacked or single
    is_stacked = "+" in stat

    if is_stacked:
        parts = stat.split("+")
        colors = {
            "PTS": COLORS["pts"],
            "AST": COLORS["ast"],
            "REB": COLORS["reb"]
        }

        # Calculate total for threshold
        totals = sum(player_df[p] for p in parts)
        threshold = round(totals.mean() - 0.5) + 0.5

        # Add stacked bars
        for i, part in enumerate(parts):
            fig.add_trace(go.Bar(
                x=list(range(len(player_df))),
                y=player_df[part],
                name=part,
                marker_color=colors.get(part, COLORS["accent"]),
                text=[f"{int(v)}<br>{part}" for v in player_df[part]],
                textposition="inside",
                textfont=dict(size=10, color="white"),
                hovertemplate=f"{part}: %{{y}}<extra></extra>"
            ))

        # Add total labels on top
        cumsum = sum(player_df[p] for p in parts)
        fig.add_trace(go.Scatter(
            x=list(range(len(player_df))),
            y=cumsum + 2,
            mode="text",
            text=[f"{int(v)}" for v in cumsum],
            textposition="top center",
            textfont=dict(size=12, color=COLORS["text"], weight="bold" if hasattr(dict, 'weight') else None),
            showlegend=False,
            hoverinfo="skip"
        ))

    else:
        # Single stat bars
        values = player_df[stat]
        threshold = round(values.mean() - 0.5) + 0.5

        colors_map = {"PTS": COLORS["pts"], "AST": COLORS["ast"], "REB": COLORS["reb"]}
        bar_color = colors_map.get(stat, COLORS["accent"])

        fig.add_trace(go.Bar(
            x=list(range(len(player_df))),
            y=values,
            marker_color=bar_color,
            text=[f"{int(v)}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate=f"{stat}: %{{y}}<extra></extra>"
        ))

        threshold = round(values.mean() - 0.5) + 0.5

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
    Output("supporting-stats", "children"),
    Input("player-dropdown", "value")
)
def update_supporting_stats(player_name):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name]
    current = player_df[player_df["SEASON"] == "2024-25"]
    if current.empty:
        current = player_df

    stats = [
        ("Minutes", current["MIN"].mean() if "MIN" in current.columns else 0),
        ("FG", f"{current['FGM'].mean():.1f}/{current['FGA'].mean():.1f}" if "FGM" in current.columns else "0/0"),
        ("3PT", f"{current['FG3M'].mean():.1f}/{current['FG3A'].mean():.1f}" if "FG3M" in current.columns else "0/0"),
        ("FT", f"{current['FTM'].mean():.1f}/{current['FTA'].mean():.1f}" if "FTM" in current.columns else "0/0"),
    ]

    return html.Div([
        html.Div("Supporting Stats", style={
            "color": COLORS["text_secondary"],
            "fontSize": "12px",
            "fontWeight": "600",
            "marginBottom": "12px",
            "textTransform": "uppercase",
            "letterSpacing": "1px"
        }),
        html.Div([
            html.Div([
                html.Div(label, style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginBottom": "4px"}),
                html.Div(f"{val:.1f}" if isinstance(val, float) else val, style={"fontWeight": "600", "fontSize": "14px"})
            ], style={"marginRight": "32px"}) for label, val in stats
        ], style={"display": "flex"})
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
                    "avg": result.get("recent_avg", 0)
                })
        except:
            pass

    colors = {"PTS": COLORS["pts"], "AST": COLORS["ast"], "REB": COLORS["reb"]}

    return html.Div([
        html.Div("PROJECTION", style={
            "color": COLORS["text_secondary"],
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "marginBottom": "16px"
        }),
        html.Div([
            html.Div([
                html.Div(pred["stat"], style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                html.Div(f"{pred['predicted']}", style={
                    "color": colors.get(pred["stat"], COLORS["accent"]),
                    "fontSize": "28px",
                    "fontWeight": "600"
                }),
                html.Div(f"avg {pred['avg']}", style={"color": COLORS["text_muted"], "fontSize": "11px"})
            ], style={
                "textAlign": "center",
                "padding": "12px",
                "backgroundColor": COLORS["bg"],
                "borderRadius": "8px",
                "flex": "1",
                "margin": "0 4px"
            }) for pred in predictions
        ], style={"display": "flex"})
    ])


@callback(
    Output("status-card", "children"),
    Input("player-dropdown", "value")
)
def update_status_card(player_name):
    if not player_name:
        return None

    try:
        status = get_player_injury_status(player_name)
        status_text = status.get("status", "ACTIVE")
        if status_text in ["UNKNOWN", "HEALTHY"]:
            status_text = "ACTIVE"
    except:
        status_text = "ACTIVE"

    status_color = {
        "ACTIVE": COLORS["hit_high"],
        "QUESTIONABLE": COLORS["hit_mid"],
        "OUT": COLORS["hit_low"]
    }.get(status_text, COLORS["hit_high"])

    player_df = DF[DF["PLAYER_NAME"] == player_name]
    games = len(player_df[player_df["SEASON"] == "2024-25"])

    return html.Div([
        html.Div("STATUS", style={
            "color": COLORS["text_secondary"],
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "marginBottom": "12px"
        }),
        html.Div([
            html.Span("●", style={"color": status_color, "marginRight": "8px", "fontSize": "12px"}),
            html.Span(status_text, style={"color": status_color, "fontWeight": "600"})
        ]),
        html.Div(f"{games} games played", style={"color": COLORS["text_muted"], "fontSize": "12px", "marginTop": "4px"})
    ])


@callback(
    Output("hit-rates-table", "children"),
    Input("player-dropdown", "value")
)
def update_hit_rates_table(player_name):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

    def calc_hit(stat, line, n):
        recent = player_df.head(n)
        if len(recent) == 0:
            return 0
        if "+" in stat:
            parts = stat.split("+")
            vals = sum(recent[p] for p in parts)
        else:
            vals = recent[stat]
        return (vals >= line).sum() / len(recent) * 100

    # Calculate averages
    avgs = {
        "PTS": player_df["PTS"].mean(),
        "AST": player_df["AST"].mean(),
        "REB": player_df["REB"].mean(),
    }

    rows = []
    for stat, avg in avgs.items():
        line = round(avg - 0.5) + 0.5
        l5 = calc_hit(stat, line, 5)
        l10 = calc_hit(stat, line, 10)
        season = calc_hit(stat, line, len(player_df))

        rows.append(html.Tr([
            html.Td(f"{stat} O{line}", style={"padding": "8px 0", "fontSize": "13px"}),
            html.Td(f"{l5:.0f}%", style={"color": get_hit_color(l5), "textAlign": "center", "fontWeight": "500"}),
            html.Td(f"{l10:.0f}%", style={"color": get_hit_color(l10), "textAlign": "center", "fontWeight": "500"}),
            html.Td(f"{season:.0f}%", style={"color": get_hit_color(season), "textAlign": "center", "fontWeight": "500"}),
        ]))

    return html.Div([
        html.Div("HIT RATES", style={
            "color": COLORS["text_secondary"],
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "marginBottom": "12px"
        }),
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Line", style={"textAlign": "left", "color": COLORS["text_muted"], "fontSize": "11px", "padding": "8px 0"}),
                    html.Th("L5", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
                    html.Th("L10", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
                    html.Th("Season", style={"textAlign": "center", "color": COLORS["text_muted"], "fontSize": "11px"}),
                ])
            ]),
            html.Tbody(rows)
        ], style={"width": "100%", "borderCollapse": "collapse"})
    ])


@callback(
    Output("splits-card", "children"),
    Input("player-dropdown", "value")
)
def update_splits_card(player_name):
    if not player_name:
        return None

    player_df = DF[DF["PLAYER_NAME"] == player_name]

    home = player_df[player_df["is_home"] == 1]
    away = player_df[player_df["is_home"] == 0]

    def get_avgs(df):
        if len(df) == 0:
            return 0, 0, 0
        return df["PTS"].mean(), df["AST"].mean(), df["REB"].mean()

    home_pts, home_ast, home_reb = get_avgs(home)
    away_pts, away_ast, away_reb = get_avgs(away)

    return html.Div([
        html.Div("SPLITS", style={
            "color": COLORS["text_secondary"],
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "marginBottom": "12px"
        }),
        html.Table([
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
                    html.Td("Home", style={"padding": "8px 0", "fontSize": "13px"}),
                    html.Td(f"{home_pts:.1f}", style={"textAlign": "center", "color": COLORS["pts"], "fontWeight": "500"}),
                    html.Td(f"{home_ast:.1f}", style={"textAlign": "center", "color": COLORS["ast"], "fontWeight": "500"}),
                    html.Td(f"{home_reb:.1f}", style={"textAlign": "center", "color": COLORS["reb"], "fontWeight": "500"}),
                ]),
                html.Tr([
                    html.Td("Away", style={"padding": "8px 0", "fontSize": "13px"}),
                    html.Td(f"{away_pts:.1f}", style={"textAlign": "center", "color": COLORS["pts"], "fontWeight": "500"}),
                    html.Td(f"{away_ast:.1f}", style={"textAlign": "center", "color": COLORS["ast"], "fontWeight": "500"}),
                    html.Td(f"{away_reb:.1f}", style={"textAlign": "center", "color": COLORS["reb"], "fontWeight": "500"}),
                ]),
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
    ])


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
