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


# Combo definitions: standard sportsbook-available props only
_COMBO_DEFS: list[tuple[list[str], str]] = [
    (["PTS", "REB"],        "Pts+Reb"),
    (["PTS", "AST"],        "Pts+Ast"),
    (["AST", "REB"],        "Ast+Reb"),
    (["PTS", "AST", "REB"], "Pts+Ast+Reb"),
]

# Hit-rate thresholds — raised from old 50%/50% coinflip levels
_OVER_MIN_HIT_RATE  = 0.52   # Over: need genuine edge, not a coin flip
_UNDER_MIN_HIT_RATE = 0.62   # Under: much higher bar (natural positive skew in NBA)

# Alt lines: lookback windows and minimum meaningful thresholds per stat
_ALT_WINDOWS       = [5, 6, 7, 8, 10, 12, 15, 17, 18, 20]
_ALT_MIN_THRESH    = {"PTS": 5, "AST": 2, "REB": 3, "FG3M": 1}
_ALT_STAT_LABELS   = {"PTS": "POINTS", "AST": "ASSISTS",
                      "REB": "REBOUNDS", "FG3M": "MADE THREES"}


def _is_qualified_player(player_name: str, player_df: "pd.DataFrame") -> tuple[bool, float]:
    """Return (qualified, avg_min_l10).

    Filters out:
    - Retired / inactive players (no game within 45 days)
    - DNP / end-of-bench players (< 15 MPG in last 10 games)
    - Players with too little current-season data (< 10 games this season)
    """
    if len(player_df) < 10:
        return False, 0.0

    # Must have played within the last 45 days
    most_recent = player_df["_date"].iloc[0]
    try:
        days_inactive = (datetime.now() - most_recent.to_pydatetime()).days
    except Exception:
        days_inactive = (datetime.now() - most_recent).days
    if days_inactive > 45:
        return False, 0.0

    # Must average at least 15 MPG over last 10 games
    recent_min = pd.to_numeric(player_df.head(10)["MIN"], errors="coerce")
    avg_min = recent_min.mean()
    if pd.isna(avg_min) or avg_min < 15:
        return False, 0.0

    # Must have at least 10 games in the current season
    if "SEASON" in player_df.columns:
        current_rows = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
        if len(current_rows) < 10:
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


def _compute_main_page_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info, availability_map, players_to_analyze, game_spreads=None, get_predictor_fn=None):
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

    # Process all players
    processed_players = set()
    for player_name in players_to_analyze:
        if player_name in processed_players:
            continue
        processed_players.add(player_name)

        is_avail, reason = availability_map.get(player_name, (True, ""))
        if not is_avail:
            continue

        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

        # ── Qualification gate: activity, minutes, current-season data ────────
        qualified, avg_min = _is_qualified_player(player_name, player_df)
        if not qualified:
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
        role     = _get_player_role(avg_min)

        # ── Use current-season data preferentially ────────────────────────────
        if "SEASON" in player_df.columns:
            cs_df = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
            recent_10 = cs_df.head(10) if len(cs_df) >= 5 else player_df.head(10)
        else:
            recent_10 = player_df.head(10)

        # Home/away splits — also prefer current season
        split_base = cs_df if ("SEASON" in player_df.columns and len(cs_df) >= 5) else player_df
        home_games = split_base[split_base["MATCHUP"].str.contains("vs.", na=False)].head(10) if "MATCHUP" in split_base.columns else split_base.head(10)
        away_games = split_base[split_base["MATCHUP"].str.contains("@",   na=False)].head(10) if "MATCHUP" in split_base.columns else split_base.head(10)

        # ── Blowout risk (role-aware) ─────────────────────────────────────────
        _spreads    = game_spreads or {}
        team_spread = _spreads.get(player_team)
        blowout_spread = abs(team_spread) if team_spread is not None else None
        blowout_risk   = blowout_spread is not None and blowout_spread >= 10

        for stat_type in ["PTS", "AST", "REB", "FG3M"]:
            if stat_type not in recent_10.columns:
                continue

            recent_stats = pd.to_numeric(recent_10[stat_type], errors="coerce").dropna()
            if len(recent_stats) < 5:
                continue

            avg_stat    = recent_stats.mean()

            if avg_stat < 1:
                continue

            # ── L5 average — primary form signal (current performance level) ──
            # Using L5 instead of L10 median so that old injury/rest games don't
            # artificially drag the line down. If a player avg 20.5 over L5 the
            # line should be ~20.5, not ~10.5 because of stale L10 data.
            l5_avg = float(recent_stats.head(5).mean()) if len(recent_stats) >= 5 else float(avg_stat)

            if stat_type == "PTS":
                line = math.floor(l5_avg) + 0.5 if l5_avg > 5 else 4.5
            elif stat_type == "FG3M":
                line = math.floor(l5_avg) + 0.5 if l5_avg > 1 else 0.5
            else:  # AST, REB
                line = math.floor(l5_avg) + 0.5 if l5_avg > 2 else 1.5

            n = len(recent_stats)

            # ── ML line blending (60% ML, 40% L5-based line) ──────────────────
            # Store ml_pred_stored for use in EV calculation later.
            ml_pred_stored: float | None = None
            if get_predictor_fn:
                try:
                    predictor = get_predictor_fn(stat_type)
                    if predictor:
                        ml_result = predictor.predict_player_game(player_name, DF)
                        ml_pred = ml_result.get(f"predicted_{stat_type.lower()}")
                        if ml_pred and float(ml_pred) > 0:
                            ml_pred_stored = float(ml_pred)
                            blended = 0.6 * ml_pred_stored + 0.4 * float(line)
                            line = math.floor(blended) + 0.5
                except Exception:
                    pass  # stay with L5-based line

            over_line  = line
            under_line = over_line + 1.0

            hits_over  = (recent_stats >= over_line).sum()
            hits_under = (recent_stats <  under_line).sum()
            hit_rate_over  = hits_over  / n
            hit_rate_under = hits_under / n

            opp_def  = DEFENSE_VS_POS[
                (DEFENSE_VS_POS["TEAM_ABBREVIATION"] == opponent) &
                (DEFENSE_VS_POS["POSITION"] == position)
            ] if not DEFENSE_VS_POS.empty else pd.DataFrame()
            rank_col = f"{stat_type}_RANK" if stat_type != "FG3M" else "3PM_RANK"
            def_rank = int(opp_def.iloc[0].get(rank_col, 15)) if not opp_def.empty else None

            game_matchup_str = (
                f"{opponent} @ {_normalize_abbr(player_team)}" if is_home_today
                else f"{_normalize_abbr(player_team)} @ {opponent}"
            )

            # ── Consistency multiplier (penalise high-variance players) ───────
            std_stat = recent_stats.std()
            cv = std_stat / avg_stat if avg_stat > 0 else 1.0
            consistency_mult = max(0.75, 1.0 - max(0.0, cv - 0.20) * 0.60)

            def _make_prop(direction, bet_line, hit_rate, hits,
                           _role=role, _blowout_risk=blowout_risk,
                           _blowout_spread=blowout_spread, _cons=consistency_mult,
                           _l5_avg=l5_avg, _model_pred=ml_pred_stored, _std=std_stat):
                # EV initially from hit_rate; will be overwritten in live-odds
                # enrichment step with true model_prob vs implied_prob.
                ev_value = calculate_ev(hit_rate)

                # ── Role-aware blowout adjustment ─────────────────────────────
                if _blowout_risk:
                    spread_factor = 0.25 if _blowout_spread >= 15 else 0.12
                    if _role in ("star", "starter"):
                        # Starters get rested → fewer minutes → OVERs suffer
                        if direction == "Over":
                            ev_value *= (1 - spread_factor)
                        else:  # Under becomes more likely for resting starters
                            ev_value *= (1 + spread_factor * 0.5)
                    else:  # rotation / bench
                        # Bench/rotation get garbage time → OVERs improve
                        if direction == "Over":
                            ev_value *= (1 + spread_factor * 0.4)
                        else:  # Under for bench in blowout = nonsensical
                            ev_value *= 0.30

                # Consistency penalty
                ev_value *= _cons

                is_lock = (
                    hit_rate >= 0.80 and n >= 5
                    and not (_blowout_risk and _role in ("star", "starter") and direction == "Over")
                )
                insight = generate_player_insight(
                    player_name=player_name, stat=stat_type, line=bet_line,
                    opponent=opponent, player_df=player_df,
                    defense_vs_pos=DEFENSE_VS_POS, is_home=is_home_today, position=position,
                )
                if _blowout_risk:
                    role_note = "starter benched" if _role in ("star", "starter") else "bench gets garbage time"
                    insight["narrative"] = (
                        f"⚠ Blowout risk ({_blowout_spread:.0f}-pt spread, {role_note}). "
                        + insight.get("narrative", "")
                    )
                if direction == "Over":
                    hr_home = (home_games[stat_type] >= bet_line).sum() / len(home_games) if not home_games.empty else 0
                    hr_away = (away_games[stat_type] >= bet_line).sum() / len(away_games) if not away_games.empty else 0
                    h_home  = (home_games[stat_type] >= bet_line).sum() if not home_games.empty else 0
                    h_away  = (away_games[stat_type] >= bet_line).sum() if not away_games.empty else 0
                else:
                    hr_home = (home_games[stat_type] < bet_line).sum() / len(home_games) if not home_games.empty else 0
                    hr_away = (away_games[stat_type] < bet_line).sum() / len(away_games) if not away_games.empty else 0
                    h_home  = (home_games[stat_type] < bet_line).sum() if not home_games.empty else 0
                    h_away  = (away_games[stat_type] < bet_line).sum() if not away_games.empty else 0
                # model_pred: ML prediction when available, else L5 avg.
                # Used in live-odds enrichment to compute true EV.
                _model_pred_val = round(_model_pred, 1) if _model_pred else round(_l5_avg, 1)
                return {
                    "player": player_name, "team": player_team, "opponent": opponent, "position": position,
                    "role": _role,
                    "stat": stat_type, "line": bet_line, "avg": round(avg_stat, 1),
                    "l5_avg": round(_l5_avg, 1),
                    "model_pred": _model_pred_val,
                    "stat_std": round(_std, 2),
                    "direction": direction,
                    "hit_rate": hit_rate, "hits": hits, "total": n,
                    "def_rank": def_rank, "is_home_today": is_home_today,
                    "ev": ev_value,
                    "model_prob": None,   # filled in live-odds enrichment
                    "implied_prob": None, # filled in live-odds enrichment
                    "edge": None,         # filled in live-odds enrichment
                    "is_lock": is_lock,
                    "is_combo": False,
                    "blowout_risk": _blowout_risk,
                    "blowout_spread": _blowout_spread,
                    "game_matchup": game_matchup_str,
                    "hit_rate_home": hr_home, "hit_rate_away": hr_away,
                    "hits_home": h_home, "hits_away": h_away,
                    "total_home": len(home_games), "total_away": len(away_games),
                    "avg_home": round(home_games[stat_type].mean(), 1) if not home_games.empty else 0,
                    "avg_away": round(away_games[stat_type].mean(), 1) if not away_games.empty else 0,
                    "insight": insight,
                }

            # Over — require genuine edge (>52%)
            if hit_rate_over > _OVER_MIN_HIT_RATE:
                props_data.append(_make_prop("Over", over_line, hit_rate_over, hits_over))

            # Under — high bar (62%) AND must make contextual sense
            under_makes_sense = not (blowout_risk and role in ("rotation", "bench"))
            if (hit_rate_under > _UNDER_MIN_HIT_RATE
                    and hit_rate_under != hit_rate_over
                    and under_makes_sense):
                props_data.append(_make_prop("Under", under_line, hit_rate_under, hits_under))

        # ── Combo props ───────────────────────────────────────────────────────
        for combo_stats, combo_label in _COMBO_DEFS:
            if not all(s in recent_10.columns for s in combo_stats):
                continue

            avgs      = {s: pd.to_numeric(recent_10[s], errors="coerce").mean() for s in combo_stats}
            total_avg = sum(avgs.values())
            if total_avg < 2:
                continue

            raw_combo = recent_10[combo_stats].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            # Median-based combo line
            line_combo = math.floor(raw_combo.median()) + 0.5
            hits_combo    = (raw_combo >= line_combo).sum()
            hit_rate_combo = hits_combo / len(raw_combo) if len(raw_combo) > 0 else 0

            if hit_rate_combo <= 0.55:
                continue

            ev_combo = calculate_ev(hit_rate_combo)

            # Role-aware blowout for combos
            if blowout_risk:
                spread_factor = 0.25 if blowout_spread >= 15 else 0.12
                if role in ("star", "starter"):
                    ev_combo *= (1 - spread_factor)
                else:
                    ev_combo *= (1 + spread_factor * 0.3)

            ev_combo *= consistency_mult   # consistency penalty applies to combos too

            is_lock_combo = (
                hit_rate_combo >= 0.80 and len(raw_combo) >= 5
                and not (blowout_risk and role in ("star", "starter"))
            )

            insight_combo = (
                ("⚠ Blowout risk. " if blowout_risk else "")
                + f"{player_name} averages "
                + " + ".join(f"{avgs[s]:.1f} {s}" for s in combo_stats)
                + f" = {total_avg:.1f} combined (line {line_combo}). "
                + f"Hit {hits_combo}/{len(raw_combo)} L{len(raw_combo)}."
            )
            props_data.append({
                "player":       player_name,
                "team":         player_team,
                "opponent":     opponent,
                "position":     position,
                "role":         role,
                "stat":         "+".join(combo_stats),
                "stat_label":   combo_label,
                "line":         line_combo,
                "avg":          round(total_avg, 1),
                "direction":    "Over",
                "hit_rate":     hit_rate_combo,
                "hits":         hits_combo,
                "total":        len(raw_combo),
                "def_rank":     None,
                "is_home_today": is_home_today,
                "ev":           ev_combo,
                "is_lock":      is_lock_combo,
                "is_combo":     True,
                "blowout_risk":   blowout_risk,
                "blowout_spread": blowout_spread,
                "game_matchup": f"{opponent} @ {player_team}" if is_home_today else f"{player_team} @ {opponent}",
                "hit_rate_home": 0, "hit_rate_away": 0,
                "hits_home": 0, "hits_away": 0,
                "total_home": 0, "total_away": 0,
                "avg_home": 0, "avg_away": 0,
                "insight": insight_combo,
                "l5_avg": round(float(raw_combo.head(5).mean()), 1) if len(raw_combo) >= 5 else round(total_avg, 1),
                "model_pred": round(total_avg, 1),
                "stat_std": round(float(raw_combo.std()), 2) if len(raw_combo) > 1 else 3.0,
                "model_prob": None, "implied_prob": None, "edge": None,
                "has_live_odds": False,
                "live_line": None, "live_over_price": None,
                "live_under_price": None, "live_bookmaker": None,
            })

    # ── Precompute per-player recent stats for enrichment ─────────────────
    # Avoids rescanning the full DF for every prop during the odds loop.
    _player_stat_cache: dict = {}
    for _prop in props_data:
        _key = (_prop["player"], _prop["stat"])
        if _key not in _player_stat_cache:
            _p_df = DF[DF["PLAYER_NAME"] == _prop["player"]].sort_values("_date", ascending=False)
            if "SEASON" in _p_df.columns:
                _cs = _p_df[_p_df["SEASON"].str.startswith("2025", na=False)]
                _r10 = _cs.head(10) if len(_cs) >= 5 else _p_df.head(10)
            else:
                _r10 = _p_df.head(10)
            _stat = _prop["stat"]
            if _stat in _r10.columns and not _r10.empty:
                _vals = pd.to_numeric(_r10[_stat], errors="coerce").dropna()
                _player_stat_cache[_key] = {
                    "vals": _vals,
                    "std":  float(_vals.std()) if len(_vals) > 1 else max(_prop.get("l5_avg", 4) * 0.25, 3.0),
                }
            else:
                _player_stat_cache[_key] = {"vals": pd.Series(dtype=float), "std": 4.0}

    # ── Enrich with live sportsbook odds ──────────────────────────────────
    # For each prop:
    #   - Overwrite line with real sportsbook line
    #   - Recalculate hit_rate at that real line (not the model-generated line)
    #   - EV = model_prob vs implied_prob (true edge, not just hit_rate × payout)
    live_odds = get_live_odds()
    for prop in props_data:
        player  = prop["player"]
        stat    = prop["stat"]
        p_odds  = live_odds.get(player) or live_odds.get(player.replace(".", "").replace("  ", " ").strip())
        s_odds  = p_odds.get(stat) if p_odds else None

        # Determine std_dev for probability calculation
        _pkey = (player, stat)
        _cached = _player_stat_cache.get(_pkey, {})
        _std = _cached.get("std") or prop.get("stat_std") or max(prop.get("l5_avg", 4) * 0.25, 3.0)
        _vals = _cached.get("vals", pd.Series(dtype=float))

        # model_pred: use ML prediction when available, else L5 avg
        _mpred = prop.get("model_pred") or prop.get("l5_avg") or prop.get("avg") or 0.0

        if s_odds:
            sb_line        = float(s_odds["line"])
            sb_over_price  = int(s_odds.get("over_price",  -110))
            sb_under_price = int(s_odds.get("under_price", -110))
            direction      = prop.get("direction", "Over")
            sb_price       = sb_over_price if direction == "Over" else sb_under_price

            prop["live_line"]        = sb_line
            prop["live_over_price"]  = sb_over_price
            prop["live_under_price"] = sb_under_price
            prop["live_bookmaker"]   = s_odds["bookmaker"]
            prop["has_live_odds"]    = True
            prop["line"]             = sb_line

            # Recalculate hit_rate against the ACTUAL sportsbook line
            if len(_vals) >= 3:
                if direction == "Over":
                    sb_hits     = int((_vals >= sb_line).sum())
                else:
                    sb_hits     = int((_vals <  sb_line).sum())
                sb_hit_rate = round(sb_hits / len(_vals), 4)
                prop["hit_rate"] = sb_hit_rate
                prop["hits"]     = sb_hits
                prop["total"]    = len(_vals)

            # EV = model's probability estimate vs sportsbook's implied probability
            model_prob = calculate_hit_probability(
                prediction=float(_mpred),
                line=sb_line,
                std_dev=_std,
                direction=direction.lower(),
                stat_type=stat,
            )
            # Implied probability from sportsbook price
            if sb_price > 0:
                implied = sb_price / (sb_price + 100)
            else:
                implied = abs(sb_price) / (abs(sb_price) + 100)

            prop["model_prob"]   = round(model_prob, 4)
            prop["implied_prob"] = round(implied, 4)
            prop["edge"]         = round(model_prob - implied, 4)
            prop["ev"]           = calculate_ev(model_prob, over_american=sb_price)

        else:
            # No live odds: use model_prob vs default -110 implied (52.4%)
            model_prob = calculate_hit_probability(
                prediction=float(_mpred),
                line=float(prop["line"]),
                std_dev=_std,
                direction=prop.get("direction", "Over").lower(),
                stat_type=stat,
            )
            prop["model_prob"]   = round(model_prob, 4)
            prop["implied_prob"] = 0.524
            prop["edge"]         = round(model_prob - 0.524, 4)
            prop["ev"]           = calculate_ev(model_prob)
            prop["has_live_odds"]    = False
            prop["live_line"]        = None
            prop["live_over_price"]  = None
            prop["live_under_price"] = None
            prop["live_bookmaker"]   = None

    # ── Quality gate — keep only high-confidence, genuine-edge props ─────
    # Locks bypass the quality gate (they already cleared 80% hit_rate).
    _MIN_MODEL_PROB = 0.58   # model must give ≥58% probability
    _MIN_EV         = 0.02   # ≥2% edge over implied (model_prob − implied ≥ 0.02)
    _MAX_PER_PLAYER = 2      # max 2 props per player to avoid flooding one hot player
    _HARD_CAP       = 25     # absolute max props shown

    quality_props = []
    for p in props_data:
        if p.get("is_lock"):
            quality_props.append(p)
            continue
        mp = p.get("model_prob") or 0.0
        ev = p.get("ev") or 0.0
        if mp >= _MIN_MODEL_PROB and ev >= _MIN_EV:
            quality_props.append(p)

    # Deduplicate: keep top _MAX_PER_PLAYER props per player (by EV)
    from collections import defaultdict as _dd
    _per_player: dict = _dd(list)
    for p in quality_props:
        _per_player[p["player"]].append(p)

    deduped: list = []
    for _pprops in _per_player.values():
        _pprops.sort(key=lambda x: (not x.get("is_lock", False), -(x.get("ev") or 0)))
        deduped.extend(_pprops[:_MAX_PER_PLAYER])

    # Final sort: LOCKs first, then by EV descending
    deduped.sort(key=lambda x: (not x.get("is_lock", False), -(x.get("ev") or 0)))
    props_data = deduped[:_HARD_CAP]

    print(f"[PropsCache] Quality gate: {len(props_data)} props (min model_prob={_MIN_MODEL_PROB}, min EV={_MIN_EV})")

    # Warn if any teams playing today have zero props (may indicate data gap)
    if game_info["has_todays_games"] and teams_today:
        teams_with_props = {p["team"] for p in props_data}
        teams_without = teams_today - teams_with_props
        if teams_without:
            print(f"[PropsCache] Teams with no qualifying props after quality gate: {teams_without}")

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

        for stat, min_thresh in _ALT_MIN_THRESH.items():
            if stat not in source_df.columns:
                continue

            stat_series = (
                pd.to_numeric(source_df[stat], errors="coerce")
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
    main_data     = _compute_main_page_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info, availability_map, players_to_analyze, game_spreads=game_spreads, get_predictor_fn=get_predictor_fn)
    callback_data = _compute_callback_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, game_info, availability_map)
    sidebar_data  = _compute_sidebar_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, PLAYERS, game_info, get_predictor_fn, availability_map=availability_map)
    # Preserve today's alt_lines across intra-day refreshes — recompute only when:
    # (a) it's a new calendar day, or (b) cache is still empty (first run / no results yet)
    today_str = datetime.now().strftime("%Y-%m-%d")
    with _cache_lock:
        existing_alt_date  = _props_cache.get("alt_lines_date")
        existing_alt_lines = _props_cache.get("alt_lines_data", [])

    if existing_alt_date == today_str and existing_alt_lines:
        alt_lines = existing_alt_lines
        print(f"[PropsCache] Preserving {len(alt_lines)} alt lines from earlier today ({today_str})")
    else:
        alt_lines = _compute_alt_lines(DF, PLAYER_POSITIONS, game_info, availability_map, players_to_analyze)

    elapsed = (datetime.now() - start).total_seconds()

    with _cache_lock:
        _props_cache = {
            "main_page_data": main_data,
            "callback_data": callback_data,
            "sidebar_data": sidebar_data,
            "alt_lines_data": alt_lines,
            "alt_lines_date": today_str,
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
