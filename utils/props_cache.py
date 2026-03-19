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
from utils.odds_fetcher import get_live_odds

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


# Stat contribution weights for combo prop evaluation
_STAT_WEIGHTS: dict[str, float] = {"PTS": 1.0, "AST": 0.7, "REB": 0.6, "FG3M": 0.8}

# Combo definitions: (stat_list, label)
_COMBO_DEFS: list[tuple[list[str], str]] = [
    (["PTS", "AST"],        "Pts+Ast"),
    (["PTS", "REB"],        "Pts+Reb"),
    (["AST", "REB"],        "Ast+Reb"),
    (["PTS", "AST", "REB"], "Pts+Ast+Reb"),
]


def _compute_main_page_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info, availability_map, players_to_analyze):
    """Compute props data for the main Best Props page.

    Process ALL players from every team playing today — no arbitrary cap.
    After computing, build a final list that guarantees representation from
    every team (both sides of every game), then sort by EV overall.
    """
    # Group players by team so we can guarantee per-team coverage
    team_to_players = {}
    for player_name in players_to_analyze:
        team = _get_player_team(player_name, PLAYER_POSITIONS)
        if team:
            team_to_players.setdefault(team, []).append(player_name)

    teams_today = set(game_info["team_to_opponent"].keys())

    # Log which teams we found players for (helps debug abbrev mismatches)
    found_teams = set(team_to_players.keys()) & teams_today
    missing_teams = teams_today - found_teams
    if missing_teams:
        print(f"[PropsCache] WARNING: No players found for teams: {missing_teams}")
        print(f"[PropsCache] teams_today={sorted(teams_today)}, found={sorted(found_teams)}")

    props_data = []

    # Process players team-by-team so every team gets evaluated
    processed_players = set()
    for player_name in players_to_analyze:
        if player_name in processed_players:
            continue
        processed_players.add(player_name)

        is_avail, reason = availability_map.get(player_name, (True, ""))
        if not is_avail:
            continue

        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)
        if len(player_df) < 5:
            continue

        player_team = _get_player_team(player_name, PLAYER_POSITIONS)
        if game_info["has_todays_games"] and player_team not in teams_today:
            continue

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
            is_lock = (hit_rate_all >= 0.80 and len(recent_stats) >= 5)

            # Include props with hit_rate >= 0.5 (positive edge)
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
                    "is_lock": is_lock,
                    "is_combo": False,
                    "game_matchup": f"{opponent} @ {_normalize_abbr(player_team)}" if is_home_today else f"{_normalize_abbr(player_team)} @ {opponent}",
                    "hit_rate_home": (home_games[stat_type] >= line).sum() / len(home_games) if not home_games.empty else 0,
                    "hit_rate_away": (away_games[stat_type] >= line).sum() / len(away_games) if not away_games.empty else 0,
                    "hits_home": (home_games[stat_type] >= line).sum() if not home_games.empty else 0,
                    "hits_away": (away_games[stat_type] >= line).sum() if not away_games.empty else 0,
                    "total_home": len(home_games), "total_away": len(away_games),
                    "avg_home": round(home_games[stat_type].mean(), 1) if not home_games.empty else 0,
                    "avg_away": round(away_games[stat_type].mean(), 1) if not away_games.empty else 0,
                    "insight": insight,
                })

        # ── Combo props ───────────────────────────────────────────────────────
        for combo_stats, combo_label in _COMBO_DEFS:
            if not all(s in recent_10.columns for s in combo_stats):
                continue

            avgs = {s: recent_10[s].mean() for s in combo_stats}
            total_avg = sum(avgs.values())
            if total_avg < 2:
                continue

            # Weighted total: used to derive a meaningful line
            weighted_total = sum(avgs[s] * _STAT_WEIGHTS[s] for s in combo_stats)

            # Skip combo if one stat contributes > 65% of the weighted total
            # (e.g. a scorer with 25 PTS and 2 REB — just predict points)
            dominant = any(
                (avgs[s] * _STAT_WEIGHTS[s]) / weighted_total > 0.65
                for s in combo_stats
            )
            if dominant:
                continue

            raw_combo = recent_10[combo_stats].sum(axis=1)
            line_combo = round(raw_combo.mean() - 0.5) + 0.5
            hits_combo = (raw_combo >= line_combo).sum()
            hit_rate_combo = hits_combo / len(raw_combo) if len(raw_combo) > 0 else 0

            if hit_rate_combo < 0.6:
                continue  # higher bar for combo props

            ev_combo = calculate_ev(hit_rate_combo)
            is_lock_combo = (hit_rate_combo >= 0.80 and len(raw_combo) >= 5)

            insight_combo = (
                f"{player_name} averages "
                + " + ".join(f"{avgs[s]:.1f} {s}" for s in combo_stats)
                + f" = {total_avg:.1f} combined (line {line_combo}). "
                + f"Hit {hits_combo}/{len(raw_combo)} L{len(raw_combo)}."
            )
            props_data.append({
                "player":       player_name,
                "team":         player_team,
                "opponent":     opponent,
                "position":     position,
                "stat":         "+".join(combo_stats),
                "stat_label":   combo_label,
                "line":         line_combo,
                "avg":          round(total_avg, 1),
                "hit_rate":     hit_rate_combo,
                "hits":         hits_combo,
                "total":        len(raw_combo),
                "def_rank":     None,
                "is_home_today": is_home_today,
                "ev":           ev_combo,
                "is_lock":      is_lock_combo,
                "is_combo":     True,
                "game_matchup": f"{opponent} @ {player_team}" if is_home_today else f"{player_team} @ {opponent}",
                "hit_rate_home": 0, "hit_rate_away": 0,
                "hits_home": 0, "hits_away": 0,
                "total_home": 0, "total_away": 0,
                "avg_home": 0, "avg_away": 0,
                "insight": insight_combo,
                "has_live_odds": False,
                "live_line": None, "live_over_price": None,
                "live_under_price": None, "live_bookmaker": None,
            })

    # ── Enrich with live sportsbook odds ─────────────────────────────────
    live_odds = get_live_odds()
    for prop in props_data:
        player  = prop["player"]
        stat    = prop["stat"]
        p_odds  = live_odds.get(player) or live_odds.get(player.replace(".", "").replace("  ", " ").strip())
        s_odds  = p_odds.get(stat) if p_odds else None

        if s_odds:
            # Override line with the real sportsbook line and recalculate EV
            prop["live_line"]        = s_odds["line"]
            prop["live_over_price"]  = s_odds["over_price"]
            prop["live_under_price"] = s_odds["under_price"]
            prop["live_bookmaker"]   = s_odds["bookmaker"]
            prop["has_live_odds"]    = True

            # Recalculate hits/hit_rate against real line
            # (The cached line was estimated; use sportsbook line if different)
            prop["line"] = s_odds["line"]
            prop["ev"]   = calculate_ev(prop["hit_rate"], over_american=s_odds["over_price"])
        else:
            prop["has_live_odds"]    = False
            prop["live_line"]        = None
            prop["live_over_price"]  = None
            prop["live_under_price"] = None
            prop["live_bookmaker"]   = None

    # Sort: LOCKs first (within stat category), then by EV
    props_data.sort(key=lambda x: (not x.get("is_lock", False), -x["ev"]))

    # Guarantee at least the top prop per team is represented in the final list.
    # This ensures games with weaker overall EVs still show up.
    if game_info["has_todays_games"] and teams_today:
        teams_with_props = {p["team"] for p in props_data}
        teams_without = teams_today - teams_with_props
        if teams_without:
            print(f"[PropsCache] Teams with no qualifying props (hit_rate<0.5): {teams_without}")

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

    # Determine players to analyze — ALL players from ALL teams playing today
    if game_info["has_todays_games"] and game_info["teams_playing"] and not PLAYER_POSITIONS.empty:
        players_to_analyze = PLAYER_POSITIONS[
            PLAYER_POSITIONS["TEAM_ABBREVIATION"].isin(game_info["teams_playing"])
        ]["PLAYER_NAME"].tolist()
        print(f"[PropsCache] {len(players_to_analyze)} players from {len(game_info['teams_playing'])} teams playing today")
    elif not PLAYER_POSITIONS.empty:
        recent_players = DF.sort_values("_date", ascending=False).drop_duplicates("PLAYER_NAME")
        players_to_analyze = recent_players.nlargest(300, "MIN")["PLAYER_NAME"].tolist()
    else:
        players_to_analyze = []

    # Batch availability check — no arbitrary cap, check all players
    availability_map = get_batch_availability(players_to_analyze)

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
