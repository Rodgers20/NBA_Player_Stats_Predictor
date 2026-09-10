# utils/props_cache.py
"""
Props Pre-computation Cache
============================
Pre-computes Best Props data in background so page renders are instant.
Called once at startup and refreshed every 30 minutes by the scheduler.

Three consumers read from this cache:
- create_best_props_page() → main page
- update_best_props_main() → callback
- create_best_props_content() → sidebar
"""

import math
import threading
from datetime import datetime

import pandas as pd

from utils.data_fetch import get_todays_games, get_upcoming_games, extract_opponent_from_matchup
from utils.injury_news import get_batch_availability
from utils.prop_calculator import calculate_ev, calculate_hit_probability
from utils.insight_generator import generate_player_insight
from utils.odds_fetcher import get_live_odds

# Thread-safe cache
_cache_lock = threading.Lock()
_props_cache = {
    "main_page_data": [],       # For create_best_props_page()
    "callback_data": [],        # For update_best_props_main()
    "sidebar_data": [],         # For create_best_props_content()
    "alt_lines_data": [],       # 100% alt lines (hit every game in streak)
    "alt_lines_date": None,     # "YYYY-MM-DD" of when alt_lines were last computed
    "parlays_data": {           # Recommended parlays built from props + alt lines
        "ml": None, "alt": [], "over": [], "pts": [], "reb": [], "ast": [],
        "combo": [], "under": [], "totals": [], "reduced": [], "defense": [], "total_count": 0
    },
    "has_todays_games": False,
    "game_matchups": [],
    "target_date": None,        # "YYYY-MM-DD" — today or tomorrow's slate
    "timestamp": None,
}


def get_cached_props() -> dict:
    """Return cached props data (instant read)."""
    with _cache_lock:
        return _props_cache.copy()


def get_parlays_cache() -> dict:
    """Return the parlays dict from the last cache refresh (instant read)."""
    with _cache_lock:
        return _props_cache.get("parlays_data", {
            "ml": None, "alt": [], "over": [], "under": [],
            "reduced": [], "defense": [], "total_count": 0
        })


# ESPN/NBA API abbreviations → internal data abbreviations
# Add any future mismatches here
_ABBR_ALIAS: dict[str, str] = {
    "SAS": "SAN",   # San Antonio: API returns SAS, data files use SAN
}


def _normalize_abbr(abbr: str) -> str:
    """Translate API team abbreviation to the one used in internal data files."""
    return _ABBR_ALIAS.get(abbr, abbr)


def _get_todays_game_info():
    """Get upcoming games info — shared by all 3 cache builders.

    Uses get_upcoming_games() which automatically falls back to tomorrow's
    slate when all of today's games have finished.
    """
    games, target_date = get_upcoming_games()
    teams_playing = []
    teams_home_away = {}
    team_to_opponent = {}
    game_matchups = []
    has_todays_games = False

    if not games.empty:
        has_todays_games = True
        for _, game in games.iterrows():
            home = _normalize_abbr(game.get("HOME_TEAM", ""))
            away = _normalize_abbr(game.get("AWAY_TEAM", ""))
            if home and away:
                game_matchups.append(f"{away} @ {home}")
                team_to_opponent[home] = away
                team_to_opponent[away] = home
            if home:
                teams_playing.append(home)
                teams_home_away[home] = "home"
            if away:
                teams_playing.append(away)
                teams_home_away[away] = "away"

    return {
        "games": games,
        "teams_playing": teams_playing,
        "teams_home_away": teams_home_away,
        "team_to_opponent": team_to_opponent,
        "game_matchups": game_matchups,
        "has_todays_games": has_todays_games,
        "target_date": target_date,
    }


def _get_player_team(player_name, player_positions_df):
    """Get a player's current team abbreviation."""
    if player_positions_df.empty:
        return ""
    pos_match = player_positions_df[player_positions_df["PLAYER_NAME"] == player_name]
    if len(pos_match) > 0:
        return str(pos_match["TEAM_ABBREVIATION"].iloc[0])
    return ""


def _get_player_position(player_name, player_positions_df):
    """Get a player's position."""
    if player_positions_df.empty:
        return "F"
    pos_match = player_positions_df[player_positions_df["PLAYER_NAME"] == player_name]
    if len(pos_match) > 0:
        return str(pos_match["POSITION"].iloc[0])
    return "F"


def _resolve_opponent(player_name, player_team, player_df, game_info):
    """Get opponent from today's games or fall back to last game log."""
    if game_info["has_todays_games"]:
        opp = game_info["team_to_opponent"].get(player_team, "")
        if opp:
            return opp

    # Fallback: most recent opponent from game log
    if not player_df.empty and "MATCHUP" in player_df.columns:
        last_matchup = player_df.iloc[0].get("MATCHUP", "")
        return extract_opponent_from_matchup(last_matchup)
    return ""


# Combo definitions: standard sportsbook-available props only
_COMBO_DEFS: list[tuple[list[str], str]] = [
    (["PTS", "REB"],        "Pts+Reb"),
    (["PTS", "AST"],        "Pts+Ast"),
    (["AST", "REB"],        "Ast+Reb"),
    (["PTS", "AST", "REB"], "Pts+Ast+Reb"),
]

# Hit-rate thresholds — include all props with any positive edge;
# quality gate below does the heavy filtering.
_OVER_MIN_HIT_RATE  = 0.40   # Over: allow props with any meaningful hit rate into pipeline

# Alt lines: lookback windows and minimum meaningful thresholds per stat
# Thresholds are set to levels that sportsbooks actually offer lines for:
#   PTS  ≥10 — books don't offer below 10.5 for any meaningful starter
#   AST  ≥3  — books rarely offer under 2.5 assists
#   REB  ≥4  — books rarely offer under 3.5 rebounds
#   FG3M ≥2  — books offer 1.5+, floor at 2 makes streak meaningful
#   BLK  ≥1  — defense parlays (1+ block per game in streak)
#   STL  ≥1  — defense parlays (1+ steal per game in streak)
_ALT_WINDOWS       = [5, 6, 7, 8, 10, 12, 15, 17, 18, 20]
_ALT_MIN_THRESH    = {"PTS": 10, "AST": 3, "REB": 4, "FG3M": 2, "BLK": 1, "STL": 1}
_VALUE_LINE_MIN    = {"PTS": 10.5, "AST": 3.5, "REB": 4.5, "FG3M": 1.5, "STL": 0.5, "BLK": 0.5}
_ALT_STAT_LABELS   = {"PTS": "POINTS", "AST": "ASSISTS", "REB": "REBOUNDS",
                      "FG3M": "MADE THREES", "BLK": "BLOCKS", "STL": "STEALS"}


def _prob_to_american(prob: float, vig: float = 0.0476) -> int:
    """Convert true probability → American odds with sportsbook vig applied.

    Books apply vig by inflating implied probability so both sides sum to >100%.
    Standard -110/-110 line creates 52.38%+52.38%=104.76% implied = 4.76% vig.

    Examples:
      70% model prob → viggged 73.3% → -275 (solid favourite)
      60% model prob → viggged 62.9% → -170 (moderate edge)
      55% model prob → viggged 57.6% → -136 (slight edge)

    Args:
        prob: True win probability (0.0–1.0)
        vig:  Vig rate (default 4.76% = standard -110 market)

    Returns: American odds integer (e.g. -150, +130)
    """
    prob    = max(0.01, min(0.99, prob))
    viggged = min(prob * (1 + vig), 0.99)
    if viggged >= 0.5:
        return int(-100 * viggged / (1 - viggged))
    return int(100 * (1 - viggged) / viggged)


def _extract_chart_window(df: "pd.DataFrame", stat_type: str) -> tuple[list[float], list[str]]:
    """Extract (values, labels) for a chart window — oldest game first (left→right).

    Labels are formatted as "(H)\\nWAS" or "(A)\\nTOR" from the MATCHUP column.
    """
    if df.empty or stat_type not in df.columns:
        return [], []
    vals_series = pd.to_numeric(df[stat_type], errors="coerce").fillna(0)
    labels: list[str] = []
    for _, row in df.iterrows():
        matchup = str(row.get("MATCHUP", ""))
        is_home = "vs." in matchup
        opp = extract_opponent_from_matchup(matchup) or "OPP"
        labels.append(f"({'H' if is_home else 'A'})\n{opp[:3]}")
    v_list = [round(float(v), 1) for v in vals_series.tolist()]
    # Reverse so oldest is on the left (same as reference image)
    return list(reversed(v_list)), list(reversed(labels))


def _extract_combo_chart_window(df: "pd.DataFrame", combo_stats: list) -> tuple[list[float], list[str]]:
    """Extract (values, labels) for a combo stat window (sums component columns).

    Same label format as _extract_chart_window so charts look identical.
    """
    if df is None or df.empty or not all(s in df.columns for s in combo_stats):
        return [], []
    combo_series = df[combo_stats].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    labels: list[str] = []
    for _, row in df.iterrows():
        matchup = str(row.get("MATCHUP", ""))
        is_home = "vs." in matchup
        opp = extract_opponent_from_matchup(matchup) or "OPP"
        labels.append(f"({'H' if is_home else 'A'})\n{opp[:3]}")
    v_list = [round(float(v), 1) for v in combo_series.tolist()]
    return list(reversed(v_list)), list(reversed(labels))


def _is_qualified_player(player_name: str, player_df: "pd.DataFrame") -> tuple[bool, float]:
    """Return (qualified, avg_min_l10).

    Filters out:
    - Retired / inactive players (no game within 45 days)
    - True bench / garbage-time players (< 10 MPG)
    - Players with no current-season history at all

    Intentionally permissive for starters/rotation players returning from injury:
    a player who played 3 days ago and averages 25+ min is clearly active.
    """
    if len(player_df) < 5:
        return False, 0.0

    # Must have played within the last 45 days
    most_recent = player_df["_date"].iloc[0]
    try:
        days_inactive = (datetime.now() - most_recent.to_pydatetime()).days
    except Exception:
        days_inactive = (datetime.now() - most_recent).days
    if days_inactive > 45:
        return False, 0.0

    # MPG check — use last 5 games so returning players aren't penalised by old
    # injury DNPs pulling down their L10 average
    recent_min = pd.to_numeric(player_df.head(5)["MIN"], errors="coerce")
    avg_min = recent_min.mean()
    if pd.isna(avg_min) or avg_min < 10:
        return False, 0.0

    # Garbage-time-only exclusion: very low minutes AND highly inconsistent
    if avg_min < 15:
        min_std = float(recent_min.std()) if len(recent_min) > 1 else 0.0
        if not pd.isna(min_std) and min_std > 12:
            return False, 0.0

    # Current-season data check — skip the count gate entirely if the player
    # played recently (≤ 14 days).  A player who just played is active regardless
    # of how many season games are in the local dataset (handles injury returns
    # like LaMelo Ball who missed large chunks of the year).
    if "SEASON" in player_df.columns and days_inactive > 14:
        current_rows = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
        if len(current_rows) < 4:
            return False, 0.0

    return True, float(avg_min)


def _get_player_role(avg_min: float) -> str:
    """Classify a player's role by average minutes played.

    Role affects blowout risk logic:
    - star / starter  → gets RESTED early in blowouts → OVER props suffer
    - rotation / bench → gets GARBAGE TIME in blowouts → OVER props benefit
    """
    if avg_min >= 30:   return "star"        # Franchise player, always in closing lineup
    elif avg_min >= 24: return "starter"     # Regular starter
    elif avg_min >= 17: return "rotation"    # Key rotation / bench starter
    else:               return "bench"       # Reserve, end-of-bench


def _compute_main_page_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info, availability_map, players_to_analyze, game_spreads=None, get_predictor_fn=None, team_injury_context=None):
    """Rank actual quoted PTS/AST/REB markets using held-out model residuals.

    A historical streak is descriptive. It cannot create a market, price, or
    calibrated probability. Unavailable models/quotes yield no recommendation.
    """
    from utils.market_evaluation import evaluate_market
    if not get_predictor_fn or not game_info.get("has_todays_games"):
        return []
    live_odds = get_live_odds()
    props = []
    prediction_date = pd.Timestamp(game_info.get("target_date") or datetime.now().date()).normalize()
    for player in dict.fromkeys(players_to_analyze):
        if not availability_map.get(player, (True, ""))[0]:
            continue
        from utils.pregame_features import prefer_identified_games
        history = prefer_identified_games(DF[DF["PLAYER_NAME"] == player]).copy()
        history["_date"] = pd.to_datetime(history["_date"], format="mixed")
        history = history[history["_date"] < prediction_date].sort_values("_date", ascending=False).drop_duplicates("_date")
        qualified, avg_min = _is_qualified_player(player, history)
        if not qualified or len(history) < 10:
            continue
        team = _get_player_team(player, PLAYER_POSITIONS)
        opponent = game_info.get("team_to_opponent", {}).get(team)
        if not opponent:
            continue
        home = game_info.get("teams_home_away", {}).get(team) == "home"
        player_odds = live_odds.get(player) or live_odds.get(player.replace(".", "").replace("  ", " ").strip()) or {}
        for stat in ("PTS", "AST", "REB"):
            quote = player_odds.get(stat)
            model = get_predictor_fn(stat)
            if not quote or model is None or stat not in history:
                continue
            residuals = getattr(model, "calibration_residuals", None)
            if residuals is None:
                continue
            try:
                result = model.predict_player_game(player, history, is_home=home, game_date=prediction_date)
                projection = result[f"predicted_{stat.lower()}"]
                line = float(quote["line"])
            except (ValueError, TypeError, KeyError):
                continue
            recent = pd.to_numeric(history[stat], errors="coerce").dropna().head(10)
            if len(recent) < 10:
                continue
            home_history = history[history["MATCHUP"].str.contains("vs.", regex=False, na=False)].head(10)
            away_history = history[history["MATCHUP"].str.contains("@", regex=False, na=False)].head(10)
            windows = {key: dict(zip(("values", "labels"), _extract_chart_window(frame, stat)))
                       for key, frame in (("l5", history.head(5)), ("l10", history.head(10)),
                                          ("l20", history.head(20)), ("home", home_history), ("away", away_history))}
            for direction in ("Over", "Under"):
                price = quote.get("over_price" if direction == "Over" else "under_price")
                evaluation = evaluate_market(projection, line, price, residuals, direction)
                if evaluation is None or evaluation["ev"] <= 0:
                    continue
                def hits(values):
                    return int((values > line).sum() if direction == "Over" else (values < line).sum())
                l5 = recent.head(5)
                prop = dict(evaluation, player=player, team=team, opponent=opponent,
                    position=_get_player_position(player, PLAYER_POSITIONS), role=_get_player_role(avg_min),
                    avg_minutes=avg_min, stat=stat, line=line, book_line=line, live_line=line,
                    projection=projection, model_pred=projection, avg=float(recent.mean()),
                    l5_avg=float(l5.mean()), stat_std=float(recent.std()), direction=direction,
                    hit_rate=hits(recent)/len(recent), hits=hits(recent), total=len(recent),
                    hit_rate_l5=hits(l5)/len(l5), hit_rate_vs_book=hits(l5)/len(l5), hits_vs_book=hits(l5),
                    is_home_today=home, is_home=home, has_live_odds=True, is_lock=False, is_combo=False,
                    live_over_price=quote.get("over_price"), live_under_price=quote.get("under_price"),
                    live_bookmaker=quote.get("bookmaker", ""), sim_book_line=None,
                    model_over_odds=None, model_under_odds=None, confidence="LOW", def_rank=None,
                    l5_values=l5.tolist(), chart_windows=windows, value_score=0.0, injury_boost="",
                    blowout_risk=abs((game_spreads or {}).get(team, 0)) >= 10,
                    blowout_spread=abs((game_spreads or {}).get(team, 0)),
                    game_matchup=f"{opponent} @ {team}" if home else f"{team} @ {opponent}",
                    insight={"narrative": "Estimated EV at the quoted line and price. Historical hit rates are descriptive; market calibration is unverified."})
                for side, frame in (("home", home_history), ("away", away_history)):
                    values = pd.to_numeric(frame[stat], errors="coerce").dropna()
                    prop["hits_" + side] = hits(values)
                    prop["total_" + side] = len(values)
                    prop["hit_rate_" + side] = hits(values)/len(values) if len(values) else 0.0
                    prop["avg_" + side] = float(values.mean()) if len(values) else 0.0
                props.append(prop)
    props.sort(key=lambda prop: -prop["ev"])
    best = {}
    for prop in props:
        best.setdefault((prop["player"], prop["stat"]), prop)
    return list(best.values())[:300]


def _compute_callback_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, game_info, availability_map):
    """Compute props for update_best_props_main() callback."""
    teams_today = set(game_info["team_to_opponent"].keys())
    has_todays_games = game_info["has_todays_games"]

    players_list = []
    player_teams = {}
    player_positions_map = {}

    for player_name in PLAYERS[:150]:
        if not PLAYER_POSITIONS.empty:
            pos_match = PLAYER_POSITIONS[PLAYER_POSITIONS["PLAYER_NAME"] == player_name]
            if len(pos_match) > 0:
                team = str(pos_match["TEAM_ABBREVIATION"].iloc[0])
                pos = str(pos_match["POSITION"].iloc[0])

                if has_todays_games and team not in teams_today:
                    continue

                players_list.append(player_name)
                player_teams[player_name] = team

                if "G" in pos:
                    p_pos = "G"
                elif "F" in pos:
                    p_pos = "F"
                elif "C" in pos:
                    p_pos = "C"
                else:
                    p_pos = "F"
                player_positions_map[player_name] = p_pos

    best_props = []

    for player_name in players_list[:100]:
        is_avail, reason = availability_map.get(player_name, (True, ""))
        if not is_avail:
            continue

        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)
        if len(player_df) < 5:
            continue

        player_team = player_teams[player_name]
        player_position = player_positions_map[player_name]
        opponent = game_info["team_to_opponent"].get(player_team, "")

        if not opponent and not player_df.empty and "MATCHUP" in player_df.columns:
            last_matchup = player_df.iloc[0].get("MATCHUP", "")
            opponent = extract_opponent_from_matchup(last_matchup)

        opp_def_rank = 15
        if not DEFENSE_VS_POS.empty and opponent:
            opp_def = DEFENSE_VS_POS[
                (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opponent) &
                (DEFENSE_VS_POS["POSITION"] == player_position)
            ]
            if len(opp_def) > 0:
                opp_def_rank = int(opp_def["PTS_RANK"].iloc[0])

        l10 = player_df.head(10)
        pts_avg = l10["PTS"].mean()
        pts_line = round(pts_avg * 0.9, 1)
        pts_hits = (l10["PTS"] > pts_line).sum()
        pts_hit_pct = pts_hits / len(l10) if len(l10) > 0 else 0
        pts_hit_display = int(pts_hit_pct * 100)

        ev_val = calculate_ev(pts_hit_pct)
        score = ev_val

        if score >= 0.15:
            confidence = "HIGH"
            conf_color = "var(--success)"
        elif score >= 0.05:
            confidence = "MED"
            conf_color = "var(--warning)"
        else:
            confidence = "LOW"
            conf_color = "var(--text-muted)"

        if ev_val > 0:
            pos_name = {"G": "guards", "F": "forwards", "C": "centers"}.get(player_position, "players")
            reason = f"vs {opponent} (#{opp_def_rank} vs {pos_name}) \u2022 {pts_hit_display}% hit rate L10"

            best_props.append({
                "player": player_name,
                "team": player_team,
                "prop": f"Over {pts_line} PTS",
                "projection": pts_avg,
                "hit_rate": pts_hit_display,
                "confidence": confidence,
                "conf_color": conf_color,
                "reason": reason,
                "score": score,
                "ev": ev_val,
                "opponent": opponent,
                "def_rank": opp_def_rank,
            })

    best_props.sort(key=lambda x: x["score"], reverse=True)
    return best_props


def _compute_sidebar_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, game_info, get_predictor_fn, availability_map=None):
    """Compute props for create_best_props_content() sidebar."""
    from utils.prop_scorer import calculate_smart_prop_score

    teams_today = set(game_info["team_to_opponent"].keys())
    has_todays_games = game_info["has_todays_games"]

    players_today = []
    player_info = {}

    for player_name in PLAYERS:
        player_df = DF[DF["PLAYER_NAME"] == player_name]
        if len(player_df) == 0:
            continue

        player_team = ""
        position = "F"
        if not PLAYER_POSITIONS.empty:
            pos_match = PLAYER_POSITIONS[PLAYER_POSITIONS["PLAYER_NAME"] == player_name]
            if len(pos_match) > 0:
                player_team = str(pos_match["TEAM_ABBREVIATION"].iloc[0])
                position = str(pos_match["POSITION"].iloc[0])

        if not player_team:
            continue

        if has_todays_games and player_team not in teams_today:
            continue

        # Skip players who are OUT or DOUBTFUL — use the pre-fetched availability map
        if availability_map is not None:
            is_avail, _ = availability_map.get(player_name, (True, ""))
            if not is_avail:
                continue

        opponent = game_info["team_to_opponent"].get(player_team, "")
        if not opponent:
            recent = player_df.sort_values("_date", ascending=False)
            if not recent.empty and "MATCHUP" in recent.columns:
                opponent = extract_opponent_from_matchup(str(recent.iloc[0].get("MATCHUP", "")))

        players_today.append(player_name)
        player_info[player_name] = {
            "team": player_team,
            "opponent": opponent,
            "position": position,
            "is_home": game_info.get("team_is_home", {}).get(
                player_team,
                "vs." in str(player_df.sort_values("_date", ascending=False).iloc[0].get("MATCHUP", ""))
                if not player_df.empty else False
            ),
        }

    if not players_today:
        players_today = PLAYERS[:50]

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

    all_props = []

    for player_name in players_today[:40]:
        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)
        recent = player_df.head(10)
        info = player_info.get(player_name, {})

        if len(recent) < 5:
            continue

        for prop_type in prop_types:
            try:
                stat_cols = [s for s in prop_type["stats"] if s in recent.columns]
                if len(stat_cols) != len(prop_type["stats"]):
                    continue

                recent_vals = recent[stat_cols].sum(axis=1)
                l10_avg = recent_vals.mean()
                line = round(l10_avg * 2) / 2

                prediction = l10_avg
                if len(prop_type["stats"]) == 1 and get_predictor_fn:
                    stat_predictor = get_predictor_fn(prop_type["stats"][0])
                    if stat_predictor:
                        try:
                            result = stat_predictor.predict_player_game(player_name, DF)
                            if "error" not in result:
                                pred_key = f"predicted_{prop_type['stats'][0].lower()}"
                                prediction = result.get(pred_key, l10_avg)
                        except Exception:
                            pass
                elif len(prop_type["stats"]) > 1 and get_predictor_fn:
                    combo_pred = 0
                    for stat in prop_type["stats"]:
                        stat_predictor = get_predictor_fn(stat)
                        if stat_predictor:
                            try:
                                result = stat_predictor.predict_player_game(player_name, DF)
                                if "error" not in result:
                                    combo_pred += result.get(f"predicted_{stat.lower()}", 0)
                            except Exception:
                                combo_pred += recent[stat].mean()
                        else:
                            combo_pred += recent[stat].mean() if stat in recent.columns else 0
                    prediction = combo_pred if combo_pred > 0 else l10_avg

                smart_score = calculate_smart_prop_score(
                    player_name=player_name,
                    stat_cols=stat_cols,
                    line=line,
                    player_df=player_df,
                    info=info,
                    defense_data=DEFENSE_VS_POS,
                    prediction=prediction,
                    injury_checker=None,
                )

                if smart_score["final_score"] >= 0.45:
                    all_props.append({
                        "player": player_name,
                        "prop_type": prop_type["name"],
                        "prop_label": prop_type["label"],
                        "prediction": prediction,
                        "line": line,
                        "hit_prob": smart_score["final_score"],
                        "l10_rate": smart_score["l10_rate"],
                        "l5_rate": smart_score["l5_rate"],
                        "positive_factors": smart_score["positive_factors"],
                        "negative_factors": smart_score["negative_factors"],
                        "role": smart_score["role"],
                        "avg_minutes": smart_score["avg_minutes"],
                        "confidence": smart_score["confidence"],
                        "opponent": info.get("opponent", ""),
                        "is_home": info.get("is_home", False),
                    })
            except Exception:
                continue

    all_props.sort(key=lambda x: x["hit_prob"], reverse=True)

    # Deduplicate: max 2 props per player
    final_props = []
    seen_players = {}
    for prop in all_props:
        player = prop["player"]
        if seen_players.get(player, 0) < 2:
            final_props.append(prop)
            seen_players[player] = seen_players.get(player, 0) + 1
        if len(final_props) >= 15:
            break

    return final_props


def _compute_alt_lines(DF, PLAYER_POSITIONS, game_info, availability_map, players_to_analyze):
    """Find player-stat combinations that cleared a threshold in 100% of last N games.

    For each player/stat, scans all windows in _ALT_WINDOWS (5→20 games) and
    picks the *longest streak* where floor(min(last_N)) >= _ALT_MIN_THRESH[stat].
    Returns a list sorted by window desc (longest streak first) then threshold desc.
    """
    alt_lines = []
    teams_today = set(game_info["team_to_opponent"].keys())

    processed: set = set()
    for player_name in players_to_analyze:
        if player_name in processed:
            continue
        processed.add(player_name)

        is_avail, _ = availability_map.get(player_name, (True, ""))
        if not is_avail:
            continue

        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

        # Same qualification gate as main props
        qualified, _ = _is_qualified_player(player_name, player_df)
        if not qualified:
            continue

        player_team = _get_player_team(player_name, PLAYER_POSITIONS)
        if game_info["has_todays_games"] and player_team not in teams_today:
            continue

        opponent   = _resolve_opponent(player_name, player_team, player_df, game_info)
        is_home    = game_info["teams_home_away"].get(player_team, "home") == "home"
        matchup    = (
            f"{opponent} @ {player_team}" if is_home
            else f"{player_team} @ {opponent}"
        )

        # Prefer current-season games for the streak check
        if "SEASON" in player_df.columns:
            cs_df = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
            source_df = cs_df if len(cs_df) >= 10 else player_df
        else:
            source_df = player_df

        # Filter out DNP / garbage-time games before computing alt-line streaks.
        # Games where MIN < 10 distort streak lengths and produce stale thresholds.
        _active_source = source_df[
            pd.to_numeric(source_df.get("MIN", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 10
        ] if "MIN" in source_df.columns else source_df

        for stat, min_thresh in _ALT_MIN_THRESH.items():
            if stat not in _active_source.columns:
                continue

            stat_series = (
                pd.to_numeric(_active_source[stat], errors="coerce")
                .dropna()
                .reset_index(drop=True)
            )
            if len(stat_series) < min(_ALT_WINDOWS):
                continue

            best_n:      int | None = None
            best_thresh: int | None = None

            for n in _ALT_WINDOWS:
                if len(stat_series) < n:
                    continue
                thresh = math.floor(stat_series.iloc[:n].min())
                if thresh >= min_thresh:
                    # Keep the longest window; tie-break on higher threshold
                    if best_n is None or n > best_n or (n == best_n and thresh > best_thresh):
                        best_n     = n
                        best_thresh = thresh

            if best_n is not None:
                # Context check: threshold must be ≥55% of player's season average.
                # Prevents "10+ PTS for a 10.5 PPG player" — technically true but
                # the line IS basically their average and books won't price it lower.
                season_avg = float(stat_series.mean())
                if season_avg > 0 and (best_thresh / season_avg) < 0.55:
                    continue  # Trivial streak — threshold is too close to their average

                # Compute role for alt parlay filtering
                _alt_min_series = pd.to_numeric(player_df.head(10)["MIN"], errors="coerce")
                _alt_avg_min = float(_alt_min_series.mean()) if not _alt_min_series.empty else 0.0
                _alt_l5_min = float(pd.to_numeric(player_df.head(5)["MIN"], errors="coerce").mean()) if len(player_df) >= 5 else _alt_avg_min
                _alt_role = "star" if _alt_avg_min >= 32 else ("starter" if _alt_avg_min >= 24 else ("rotation" if _alt_avg_min >= 17 else "bench"))

                alt_lines.append({
                    "player":     player_name,
                    "team":       player_team,
                    "opponent":   opponent,
                    "game_matchup": matchup,
                    "stat":       stat,
                    "stat_label": _ALT_STAT_LABELS[stat],
                    "threshold":  best_thresh,
                    "window":     best_n,
                    "trend":      f"{best_n}/L{best_n}",
                    "role":       _alt_role,
                    "l5_min_avg": round(_alt_l5_min, 1),
                })

    # Sort: longest streak → highest threshold → player name
    alt_lines.sort(key=lambda x: (-x["window"], -x["threshold"], x["player"]))
    return alt_lines


def refresh_props_cache(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, get_predictor_fn=None):
    """
    Pre-compute all Best Props data. Called at startup and by scheduler.

    Args:
        DF: Global player game logs DataFrame
        PLAYER_POSITIONS: Player positions DataFrame
        DEFENSE_VS_POS: Defense vs position DataFrame
        PLAYERS: List of player names
        get_predictor_fn: Function to get ML predictor models (lazy loaded)
    """
    global _props_cache

    print("[PropsCache] Refreshing props cache...")
    start = datetime.now()

    # Get today's game info (single NBA API call)
    game_info = _get_todays_game_info()

    # Determine players to analyze — ALL players from ALL teams playing today
    if game_info["has_todays_games"] and game_info["teams_playing"] and not PLAYER_POSITIONS.empty:
        players_to_analyze = PLAYER_POSITIONS[
            PLAYER_POSITIONS["TEAM_ABBREVIATION"].isin(game_info["teams_playing"])
        ]["PLAYER_NAME"].tolist()
        print(f"[PropsCache] {len(players_to_analyze)} players from {len(game_info['teams_playing'])} teams playing today")
    elif not PLAYER_POSITIONS.empty:
        recent_players = DF.sort_values("_date", ascending=False).drop_duplicates("PLAYER_NAME").copy()
        recent_players["_min_numeric"] = pd.to_numeric(recent_players["MIN"], errors="coerce").fillna(0)
        players_to_analyze = recent_players.nlargest(300, "_min_numeric")["PLAYER_NAME"].tolist()
    else:
        players_to_analyze = []

    # Batch availability check — no arbitrary cap, check all players
    availability_map = get_batch_availability(players_to_analyze)

    # ── Override stale ESPN injury data ──────────────────────────────────────
    # If ESPN marks a player as OUT/DOUBTFUL but they have a game in our data
    # within the last 45 days (same window as _is_qualified_player), trust the
    # game logs over the injury report.  This prevents stale ESPN data from
    # silently excluding active players like LaMelo Ball who returned from injury.
    _now_dt = datetime.now()
    for _pname in players_to_analyze:
        _is_avail, _reason = availability_map.get(_pname, (True, ""))
        if _is_avail:
            continue  # already available — nothing to override
        _p_df = DF[DF["PLAYER_NAME"] == _pname]
        if _p_df.empty or "_date" not in _p_df.columns:
            continue
        _last_date = _p_df["_date"].max()
        try:
            _days_since = (_now_dt - _last_date.to_pydatetime()).days
        except Exception:
            try:
                _days_since = (_now_dt - _last_date).days
            except Exception:
                continue
        if _days_since <= 45:
            print(
                f"[PropsCache] AvailabilityOverride: {_pname} ESPN={_reason!r} "
                f"but played {_days_since}d ago → marking ACTIVE"
            )
            availability_map[_pname] = (True, "ACTIVE (recent game data overrides ESPN)")

    # Build injury context: find teams with OUT starters and quantify missing usage
    team_injury_context: dict[str, dict] = {}
    for _pname in players_to_analyze:
        _is_avail, _reason = availability_map.get(_pname, (True, ""))
        if _is_avail:
            continue
        _pteam = _get_player_team(_pname, PLAYER_POSITIONS)
        if not _pteam:
            continue
        _p_df = DF[DF["PLAYER_NAME"] == _pname].sort_values("_date", ascending=False)
        if _p_df.empty:
            continue
        if "SEASON" in _p_df.columns:
            _cs = _p_df[_p_df["SEASON"].str.startswith("2025", na=False)]
            _r = _cs.head(10) if len(_cs) >= 5 else _p_df.head(10)
        else:
            _r = _p_df.head(10)
        if "PTS" not in _r.columns or "MIN" not in _r.columns:
            continue
        _avg_pts = float(pd.to_numeric(_r["PTS"], errors="coerce").mean() or 0)
        _avg_min = float(pd.to_numeric(_r["MIN"], errors="coerce").mean() or 0)
        # Only count meaningful contributors (starter-level: 12+ PPG, 24+ MPG)
        if _avg_pts >= 12 and _avg_min >= 24:
            _ctx = team_injury_context.setdefault(_pteam, {"out_players": [], "missing_pts": 0.0})
            _ctx["out_players"].append(_pname)
            _ctx["missing_pts"] += _avg_pts
            print(f"[PropsCache] InjuryBoost: {_pname} ({_pteam}) OUT — {_avg_pts:.1f} PPG redistributed")

    # Build game spreads dict for blowout risk detection
    game_spreads: dict = {}
    try:
        from utils.odds_fetcher import get_game_odds
        raw_odds = get_game_odds()
        for _key, _odds in raw_odds.items():
            _spread = (_odds.get("spread") or {})
            _home_line = _spread.get("home_line")
            if _home_line is not None:
                _home = _odds.get("home_team", "")
                _away = _odds.get("away_team", "")
                game_spreads[_home] = float(_home_line)        # negative = home favored
                game_spreads[_away] = float(_home_line) * -1   # positive = away underdog
    except Exception as _e:
        print(f"[PropsCache] Could not fetch game spreads for blowout risk: {_e}")

    # Compute all 4 data sets
    main_data     = _compute_main_page_props(
        DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info,
        availability_map, players_to_analyze,
        game_spreads=game_spreads,
        get_predictor_fn=get_predictor_fn,
        team_injury_context=team_injury_context,
    )
    callback_data = [dict(p, prop=f"{p['direction']} {p['line']} {p['stat']}",
                          projection=p['model_pred'], score=p['ev'],
                          hit_rate=round(p['hit_rate'] * 100), conf_color="var(--text-muted)",
                          reason="Estimated EV at a live sportsbook quote") for p in main_data]
    sidebar_data = [dict(p, prop_type=p['stat'], prop_label=p['stat'],
                         prediction=p['model_pred'], hit_prob=p['model_prob'],
                         l10_rate=p['hit_rate'], l5_rate=p['hit_rate_l5'],
                         positive_factors=[], negative_factors=[]) for p in main_data[:15]]
    # Unquoted alternate lines and unvalidated joint probabilities cannot be
    # presented as actionable bets. Keep the cache schema for existing views.
    today_str = datetime.now().strftime("%Y-%m-%d")
    alt_lines = []
    parlays_data = {key: [] for key in ("over", "pts", "reb", "ast", "combo", "ml",
                    "spread", "totals", "alt_over", "reduced", "alt", "under", "defense")}
    parlays_data["total_count"] = 0
    parlays_data["status"] = "Joint probabilities and alternate-line prices are not validated"

    elapsed = (datetime.now() - start).total_seconds()

    with _cache_lock:
        _props_cache = {
            "main_page_data": main_data,
            "callback_data": callback_data,
            "sidebar_data": sidebar_data,
            "alt_lines_data": alt_lines,
            "alt_lines_date": today_str,
            "parlays_data": parlays_data,
            "has_todays_games": game_info["has_todays_games"],
            "game_matchups": game_info["game_matchups"],
            "teams_today": set(game_info["team_to_opponent"].keys()),
            "target_date": game_info.get("target_date"),
            "timestamp": datetime.now(),
        }

    print(f"[PropsCache] Cache warmed in {elapsed:.1f}s — {len(main_data)} main props, {len(callback_data)} callback props, {len(sidebar_data)} sidebar props, {len(alt_lines)} alt lines")
    if not main_data:
        print(f"[PropsCache] WARNING: 0 main props generated. has_todays_games={game_info['has_todays_games']}, "
              f"teams_playing={game_info['teams_playing']}, players_to_analyze={len(players_to_analyze)}, "
              f"DF_rows={len(DF)}, PLAYER_POSITIONS_rows={len(PLAYER_POSITIONS)}")
