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

import threading
from datetime import datetime

import pandas as pd

from utils.data_fetch import get_todays_games, get_upcoming_games, extract_opponent_from_matchup
from utils.injury_news import get_batch_availability
from utils.prop_calculator import calculate_ev
from utils.insight_generator import generate_player_insight

# Thread-safe cache
_cache_lock = threading.Lock()
_props_cache = {
    "main_page_data": [],       # For create_best_props_page()
    "callback_data": [],        # For update_best_props_main()
    "sidebar_data": [],         # For create_best_props_content()
    "has_todays_games": False,
    "game_matchups": [],
    "target_date": None,        # "YYYY-MM-DD" — today or tomorrow's slate
    "timestamp": None,
}


def get_cached_props() -> dict:
    """Return cached props data (instant read)."""
    with _cache_lock:
        return _props_cache.copy()


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
            home = game.get("HOME_TEAM", "")
            away = game.get("AWAY_TEAM", "")
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


def _compute_main_page_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info, availability_map, players_to_analyze):
    """Compute props data for the main Best Props page."""
    props_data = []

    for player_name in players_to_analyze[:100]:
        is_avail, reason = availability_map.get(player_name, (True, ""))
        if not is_avail:
            continue

        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)
        if len(player_df) < 5:
            continue

        player_team = _get_player_team(player_name, PLAYER_POSITIONS)
        opponent = _resolve_opponent(player_name, player_team, player_df, game_info)
        if not opponent:
            continue

        is_home_today = game_info["teams_home_away"].get(player_team, "home") == "home"
        if not game_info["has_todays_games"] and not player_df.empty:
            last_matchup = player_df.iloc[0].get("MATCHUP", "")
            is_home_today = "vs." in str(last_matchup)

        position = _get_player_position(player_name, PLAYER_POSITIONS)
        recent_10 = player_df.head(10)

        home_games = player_df[player_df["MATCHUP"].str.contains("vs.", na=False)].head(10) if "MATCHUP" in player_df.columns else player_df.head(10)
        away_games = player_df[player_df["MATCHUP"].str.contains("@", na=False)].head(10) if "MATCHUP" in player_df.columns else player_df.head(10)

        for stat_type in ["PTS", "AST", "REB", "FG3M"]:
            if stat_type not in recent_10.columns:
                continue

            recent_stats = recent_10[stat_type]
            avg_stat = recent_stats.mean()
            if avg_stat < 1:
                continue

            if stat_type == "PTS":
                line = round(avg_stat - 0.5) + 0.5 if avg_stat > 5 else 4.5
            elif stat_type == "FG3M":
                line = round(avg_stat - 0.5) + 0.5 if avg_stat > 1 else 0.5
            else:
                line = round(avg_stat - 0.5) + 0.5 if avg_stat > 2 else 1.5

            hits_all = (recent_stats >= line).sum()
            hit_rate_all = hits_all / len(recent_stats) if len(recent_stats) > 0 else 0

            opp_def = DEFENSE_VS_POS[
                (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opponent) & (DEFENSE_VS_POS["POSITION"] == position)
            ] if not DEFENSE_VS_POS.empty else pd.DataFrame()
            rank_col = f"{stat_type}_RANK" if stat_type != "FG3M" else "3PM_RANK"
            def_rank = int(opp_def.iloc[0].get(rank_col, 15)) if not opp_def.empty else None

            ev_value = calculate_ev(hit_rate_all)

            if hit_rate_all >= 0.5:
                insight = generate_player_insight(
                    player_name=player_name,
                    stat=stat_type,
                    line=line,
                    opponent=opponent,
                    player_df=player_df,
                    defense_vs_pos=DEFENSE_VS_POS,
                    is_home=is_home_today,
                    position=position,
                )
                props_data.append({
                    "player": player_name, "team": player_team, "opponent": opponent, "position": position,
                    "stat": stat_type, "line": line, "avg": round(avg_stat, 1),
                    "hit_rate": hit_rate_all, "hits": hits_all, "total": len(recent_stats),
                    "def_rank": def_rank, "is_home_today": is_home_today,
                    "ev": ev_value,
                    "game_matchup": f"{opponent} @ {player_team}" if is_home_today else f"{player_team} @ {opponent}",
                    "hit_rate_home": (home_games[stat_type] >= line).sum() / len(home_games) if not home_games.empty else 0,
                    "hit_rate_away": (away_games[stat_type] >= line).sum() / len(away_games) if not away_games.empty else 0,
                    "hits_home": (home_games[stat_type] >= line).sum() if not home_games.empty else 0,
                    "hits_away": (away_games[stat_type] >= line).sum() if not away_games.empty else 0,
                    "total_home": len(home_games), "total_away": len(away_games),
                    "avg_home": round(home_games[stat_type].mean(), 1) if not home_games.empty else 0,
                    "avg_away": round(away_games[stat_type].mean(), 1) if not away_games.empty else 0,
                    "insight": insight,
                })

    props_data.sort(key=lambda x: x["ev"], reverse=True)
    return props_data


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


def _compute_sidebar_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, game_info, get_predictor_fn):
    """Compute props for create_best_props_content() sidebar."""
    from utils.prop_scorer import calculate_smart_prop_score
    from utils.injury_news import get_player_injury_status

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

        try:
            injury_status = get_player_injury_status(player_name)
            if injury_status.get("status") == "OUT":
                continue
        except Exception:
            pass

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

    # Determine players to analyze
    if game_info["has_todays_games"] and game_info["teams_playing"] and not PLAYER_POSITIONS.empty:
        players_to_analyze = PLAYER_POSITIONS[
            PLAYER_POSITIONS["TEAM_ABBREVIATION"].isin(game_info["teams_playing"])
        ]["PLAYER_NAME"].tolist()
    elif not PLAYER_POSITIONS.empty:
        recent_players = DF.sort_values("_date", ascending=False).drop_duplicates("PLAYER_NAME")
        players_to_analyze = recent_players.nlargest(150, "MIN")["PLAYER_NAME"].tolist()
    else:
        players_to_analyze = []

    # Single batch availability check (uses cached injury news)
    availability_map = get_batch_availability(players_to_analyze[:100])

    # Compute all 3 data sets
    main_data = _compute_main_page_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info, availability_map, players_to_analyze)
    callback_data = _compute_callback_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, game_info, availability_map)
    sidebar_data = _compute_sidebar_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, game_info, get_predictor_fn)

    elapsed = (datetime.now() - start).total_seconds()

    with _cache_lock:
        _props_cache = {
            "main_page_data": main_data,
            "callback_data": callback_data,
            "sidebar_data": sidebar_data,
            "has_todays_games": game_info["has_todays_games"],
            "game_matchups": game_info["game_matchups"],
            "teams_today": set(game_info["team_to_opponent"].keys()),
            "target_date": game_info.get("target_date"),
            "timestamp": datetime.now(),
        }

    print(f"[PropsCache] Cache warmed in {elapsed:.1f}s — {len(main_data)} main props, {len(callback_data)} callback props, {len(sidebar_data)} sidebar props")
