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
        "ml": None, "alt": [], "over": [], "under": [],
        "reduced": [], "defense": [], "total_count": 0
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
        # Lower threshold to 3 so injury-return players (e.g. LaMelo Ball with
        # only 4-6 logged games back) use their current-season data rather than
        # being silently mixed with last season's numbers.
        if "SEASON" in player_df.columns:
            cs_df = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
            recent_10 = cs_df.head(10) if len(cs_df) >= 3 else player_df.head(10)
        else:
            recent_10 = player_df.head(10)

        # Home/away splits — also prefer current season (threshold matches recent_10)
        split_base = cs_df if ("SEASON" in player_df.columns and len(cs_df) >= 3) else player_df
        home_games = split_base[split_base["MATCHUP"].str.contains("vs.", na=False)].head(10) if "MATCHUP" in split_base.columns else split_base.head(10)
        away_games = split_base[split_base["MATCHUP"].str.contains("@",   na=False)].head(10) if "MATCHUP" in split_base.columns else split_base.head(10)

        # ── Blowout risk (role-aware) ─────────────────────────────────────────
        _spreads    = game_spreads or {}
        team_spread = _spreads.get(player_team)
        blowout_spread = abs(team_spread) if team_spread is not None else None
        blowout_risk   = blowout_spread is not None and blowout_spread >= 10

        # Per-stat minimum average to qualify for a prop line
        _STAT_MIN_AVG = {
            "PTS": 1.0, "AST": 1.0, "REB": 1.0,
            "FG3M": 0.5, "STL": 0.5, "BLK": 0.3,
        }

        for stat_type in ["PTS", "AST", "REB", "FG3M", "STL", "BLK"]:
            if stat_type not in recent_10.columns:
                continue

            recent_stats = pd.to_numeric(recent_10[stat_type], errors="coerce").dropna()
            if len(recent_stats) < 5:
                continue

            avg_stat    = recent_stats.mean()

            if avg_stat < _STAT_MIN_AVG.get(stat_type, 1.0):
                continue

            # ── Recency-weighted L5 average ───────────────────────────────────
            # Filter out injury/rest games (< 20 min) before computing L5 so that
            # a player who sat out or played 8 minutes doesn't drag down the projection.
            # Fall back to unfiltered if fewer than 5 active games available.
            if "MIN" in recent_10.columns:
                _min_series = pd.to_numeric(recent_10["MIN"], errors="coerce").fillna(0)
                _active_mask = _min_series >= 20
                _active_stats = pd.to_numeric(
                    recent_10.loc[_active_mask, stat_type], errors="coerce"
                ).dropna()
            else:
                _active_stats = recent_stats

            _l5_source = _active_stats.head(5) if len(_active_stats) >= 5 else recent_stats.head(5)

            # Store raw L5 game values (most-recent first) for bar chart rendering
            _l5_raw_values: list[float] = [round(float(v), 1) for v in _l5_source.values[:5]]

            # ── Multi-window chart data ───────────────────────────────────────
            # Build per-window (values, labels) for the expandable bar chart.
            # L5 uses recent_10 rows that had active minutes; L10/L20 use full history.
            _split_base = (
                player_df[player_df["SEASON"].str.startswith("2025", na=False)]
                if "SEASON" in player_df.columns and
                   len(player_df[player_df["SEASON"].str.startswith("2025", na=False)]) >= 3
                else player_df
            )
            _l20_df = _split_base.head(20)
            _cw_l5_v,  _cw_l5_l  = _extract_chart_window(recent_10.head(5),  stat_type)
            _cw_l10_v, _cw_l10_l = _extract_chart_window(recent_10,           stat_type)
            _cw_l20_v, _cw_l20_l = _extract_chart_window(_l20_df,             stat_type)
            _cw_home_v, _cw_home_l = _extract_chart_window(home_games.head(10), stat_type)
            _cw_away_v, _cw_away_l = _extract_chart_window(away_games.head(10), stat_type)
            _chart_windows: dict = {
                "l5":   {"values": _cw_l5_v,   "labels": _cw_l5_l},
                "l10":  {"values": _cw_l10_v,  "labels": _cw_l10_l},
                "l20":  {"values": _cw_l20_v,  "labels": _cw_l20_l},
                "home": {"values": _cw_home_v, "labels": _cw_home_l},
                "away": {"values": _cw_away_v, "labels": _cw_away_l},
            }

            # Weight recent games more heavily so hot/cold streaks dominate:
            #   Game 1 (most recent): weight 3 | Game 2: weight 3 | Game 3: weight 2
            #   Game 4: weight 1 | Game 5: weight 1
            # Example: Ball last 5 active = [35.5, 35.5, 15.7, 15.7, 15.7]
            #   Simple mean = 23.6; Weighted = 28.2 (recency-dominated)
            if len(_l5_source) >= 5:
                _r5 = _l5_source.values[:5]  # [game0=most_recent, ...]
                _weights = [3, 3, 2, 1, 1]   # sum = 10
                l5_avg = float(
                    sum(_r5[i] * _weights[i] for i in range(5)) / sum(_weights)
                )
            else:
                l5_avg = float(avg_stat)

            # ── Season-average sanity floor ───────────────────────────────────
            # If injury games dragged L5 below 65% of season avg, floor it.
            # Prevents Ball showing 13 pts when he averages 22 on the season.
            _season_stats = pd.to_numeric(player_df[stat_type], errors="coerce").dropna()
            if len(_season_stats) >= 10:
                _season_avg = float(_season_stats.mean())
                _floor = _season_avg * 0.65
                if l5_avg < _floor:
                    l5_avg = _floor

            n = len(recent_stats)

            # ── Opponent defense rank (computed early — needed for projection) ─
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

            # ── Injury usage boost ────────────────────────────────────────────
            _injury_boost_note: str = ""
            if team_injury_context and player_team in team_injury_context and stat_type in ("PTS", "AST"):
                _ctx = team_injury_context[player_team]
                _missing = _ctx["missing_pts"]
                _out_names = ", ".join(_ctx["out_players"][:2])
                if role == "star":
                    _bfactor = 1.0 + min(_missing * 0.10 / max(l5_avg, 8), 0.25)
                elif role == "starter":
                    _bfactor = 1.0 + min(_missing * 0.08 / max(l5_avg, 8), 0.20)
                elif role == "rotation":
                    _bfactor = 1.0 + min(_missing * 0.05 / max(l5_avg, 6), 0.12)
                else:
                    _bfactor = 1.0
                if _bfactor > 1.02:
                    l5_avg = l5_avg * _bfactor
                    _injury_boost_note = f"⬆ {_out_names} OUT"

            # ── Contextual projection ─────────────────────────────────────────
            # Build a realistic projection using L5 avg as base, then adjust
            # for opponent defense, home/away splits, and ML model when available.
            # Books typically set lines ~1-2pts above L5 avg; our value line sits
            # at 92% of our contextual projection for genuine hit-rate value.
            proj = l5_avg

            # 1. Opponent defense adjustment (rank 1=best defense, 30=worst)
            #    Weak defense (rank 20-30) → boost proj; strong defense → reduce
            if def_rank is not None:
                def_factor = max(0.92, min(1.08, 1.0 + (def_rank - 15) * 0.006))
                proj = proj * def_factor

            # 2. Home/Away split (40% weight toward actual split avg)
            _ha_stat = pd.to_numeric(
                (home_games[stat_type] if is_home_today else away_games[stat_type]),
                errors="coerce"
            ) if stat_type in (home_games.columns if is_home_today else away_games.columns) else pd.Series(dtype=float)
            if len(_ha_stat.dropna()) >= 3:
                _ha_avg = float(_ha_stat.mean())
                proj = 0.60 * proj + 0.40 * _ha_avg

            # 3. ML model blend (when available): 55% ML, 45% contextual
            ml_pred_stored: float | None = None
            if get_predictor_fn:
                try:
                    predictor = get_predictor_fn(stat_type)
                    if predictor:
                        ml_result = predictor.predict_player_game(player_name, DF)
                        ml_pred = ml_result.get(f"predicted_{stat_type.lower()}")
                        if ml_pred and float(ml_pred) > 0:
                            ml_pred_stored = float(ml_pred)
                            raw_blend = 0.55 * ml_pred_stored + 0.45 * proj
                            # Clamp: ML model can't drag projection more than 25% below
                            # or 30% above the L5 average. Prevents stale/biased ML
                            # models from producing absurd lines (e.g. 17.5 for a 26.5 avg).
                            _ml_clamp_lo = l5_avg * 0.75
                            _ml_clamp_hi = l5_avg * 1.30
                            proj = max(_ml_clamp_lo, min(_ml_clamp_hi, raw_blend))
                except Exception:
                    pass

            # 4. Our displayed value line — 20th-percentile of actual L5 game values.
            #    By definition ≥80% of the last 5 games will beat this line, so the
            #    displayed hit-rate is always meaningful rather than random noise.
            #    Floor at 75% of l5_avg so high-variance players still get a real line.
            if len(_l5_source) >= 3:
                _l5_series = pd.to_numeric(pd.Series(list(_l5_source.values[:5])), errors="coerce").dropna()
                _p20 = float(_l5_series.quantile(0.20)) if len(_l5_series) >= 2 else l5_avg * 0.82
                raw_line = math.floor(max(_p20, l5_avg * 0.70) / 0.5) * 0.5
            else:
                raw_line = math.floor(l5_avg * 0.82 / 0.5) * 0.5
            line = max(_VALUE_LINE_MIN.get(stat_type, 0.5), raw_line)

            # 5. Simulated book line — set at l5_avg (books typically use recent average
            #    as their baseline; no artificial buffer that inflates our target line)
            sim_book_line = math.floor(l5_avg / 0.5) * 0.5

            over_line  = line
            under_line = over_line + 1.0

            hits_over  = (recent_stats >= over_line).sum()
            hits_under = (recent_stats <  under_line).sum()
            hit_rate_over  = hits_over  / n
            hit_rate_under = hits_under / n

            # ── Consistency multiplier (penalise high-variance players) ───────
            std_stat = recent_stats.std()
            cv = std_stat / avg_stat if avg_stat > 0 else 1.0
            consistency_mult = max(0.75, 1.0 - max(0.0, cv - 0.20) * 0.60)

            def _make_prop(direction, bet_line, hit_rate, hits,
                           _role=role, _blowout_risk=blowout_risk,
                           _blowout_spread=blowout_spread, _cons=consistency_mult,
                           _l5_avg=l5_avg, _model_pred=ml_pred_stored, _std=std_stat,
                           _proj=proj, _sim_book=sim_book_line,
                           _l5_vals=_l5_raw_values,
                           _chart_wins=_chart_windows):
                # EV initially from hit_rate; will be overwritten in live-odds
                # enrichment step with true model_prob vs implied_prob.
                ev_value = calculate_ev(hit_rate)

                # L5 hit rate — most recent 5 games against this specific line
                if _l5_vals:
                    _l5_h = sum(1 for v in _l5_vals if (v >= bet_line if direction == "Over" else v < bet_line))
                    _hit_rate_l5 = round(_l5_h / len(_l5_vals), 4)
                else:
                    _hit_rate_l5 = hit_rate

                # ── Role-aware blowout adjustment ─────────────────────────────
                if _blowout_risk:
                    spread_factor = 0.25 if _blowout_spread >= 15 else 0.12
                    if _role in ("star", "starter"):
                        # Starters get rested → fewer minutes → OVERs suffer
                        ev_value *= (1 - spread_factor)
                    else:  # rotation / bench
                        # Bench/rotation get garbage time → OVERs improve
                        ev_value *= (1 + spread_factor * 0.4)

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
                hr_home = (home_games[stat_type] >= bet_line).sum() / len(home_games) if not home_games.empty else 0
                hr_away = (away_games[stat_type] >= bet_line).sum() / len(away_games) if not away_games.empty else 0
                h_home  = (home_games[stat_type] >= bet_line).sum() if not home_games.empty else 0
                h_away  = (away_games[stat_type] >= bet_line).sum() if not away_games.empty else 0
                # model_pred: contextual projection (opponent/home-away/ML adjusted).
                # Used in live-odds enrichment to compute true EV, and shown on card.
                _model_pred_val = round(_model_pred, 1) if _model_pred else round(_proj, 1)
                return {
                    "player": player_name, "team": player_team, "opponent": opponent, "position": position,
                    "role": _role,
                    "stat": stat_type, "line": bet_line, "avg": round(avg_stat, 1),
                    "l5_avg": round(_l5_avg, 1),
                    "projection": round(_proj, 1),   # contextual projection (shown on card)
                    "sim_book_line": round(_sim_book, 1),  # simulated book line
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
                    "value_score": round(bet_line / max(_l5_avg, 0.1), 3),
                    "injury_boost": _injury_boost_note,
                    "l5_values": list(_l5_vals),      # raw per-game values for bar chart
                    "chart_windows": dict(_chart_wins), # multi-window chart data
                    "hit_rate_vs_book": None,           # filled in live-odds enrichment
                    "hits_vs_book": None,
                    "hit_rate_l5": _hit_rate_l5,        # L5-specific hit rate vs this line
                }

            # Over — require genuine edge (>52%)
            if hit_rate_over > _OVER_MIN_HIT_RATE:
                props_data.append(_make_prop("Over", over_line, hit_rate_over, hits_over))

        # ── Combo props (PTS+REB, PTS+AST, AST+REB, PTS+AST+REB) ───────────────
        # Generate ALL combos for every eligible player — quality gate handles
        # filtering.  Pre-generation threshold removed so players with strong
        # recent form (3+ of last 5) are not silently discarded.
        # Minimum average each component stat must meet for a combo to be relevant.
        # This prevents nonsensical props like PTS+AST for a center averaging 0.8 AST.
        _COMBO_STAT_MIN: dict[str, float] = {"PTS": 5.0, "REB": 3.0, "AST": 2.0}

        for combo_stats, combo_label in _COMBO_DEFS:
            if not all(s in recent_10.columns for s in combo_stats):
                continue

            avgs      = {s: pd.to_numeric(recent_10[s], errors="coerce").mean() for s in combo_stats}
            total_avg = sum(avgs.values())
            if total_avg < 2:
                continue

            # Skip combo if the player doesn't meaningfully contribute to every component
            if any(avgs[s] < _COMBO_STAT_MIN.get(s, 0.0) for s in combo_stats):
                continue

            raw_combo = recent_10[combo_stats].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            l5_combo  = raw_combo.head(5)

            # Line = 20th-percentile of L5 values (same logic as individual stats)
            # → ≥80% of last 5 games beat this line; floor at 70% of L5 avg
            if len(l5_combo) >= 2:
                _l5_combo_series = pd.to_numeric(l5_combo, errors="coerce").dropna()
                _cp20   = float(_l5_combo_series.quantile(0.20)) if len(_l5_combo_series) >= 2 else total_avg * 0.82
                _l5_combo_avg = float(l5_combo.mean()) if len(l5_combo) > 0 else total_avg
                line_combo = math.floor(max(_cp20, _l5_combo_avg * 0.70) / 0.5) * 0.5
            else:
                line_combo = math.floor(total_avg * 0.82 / 0.5) * 0.5
            line_combo = max(2.5, line_combo)   # absolute floor — no trivial combo lines

            hits_combo     = (raw_combo >= line_combo).sum()
            hit_rate_combo = hits_combo / len(raw_combo) if len(raw_combo) > 0 else 0

            # L5-specific hit rate — primary quality gate signal
            l5_hits_combo   = int((l5_combo >= line_combo).sum()) if len(l5_combo) > 0 else 0
            hit_rate_l5_combo = round(l5_hits_combo / len(l5_combo), 4) if len(l5_combo) > 0 else 0.0

            # Only skip if player literally never hits the combo line (< 1/5 in L5)
            if hit_rate_l5_combo < 0.20:
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
            # Build combo chart windows using the combo-aware helper
            _split_base_combo = (
                player_df[player_df["SEASON"].str.startswith("2025", na=False)]
                if "SEASON" in player_df.columns and
                   len(player_df[player_df["SEASON"].str.startswith("2025", na=False)]) >= 3
                else player_df
            )
            _l20_combo = _split_base_combo.head(20)
            _ccw_l5_v,  _ccw_l5_l  = _extract_combo_chart_window(recent_10.head(5),     combo_stats)
            _ccw_l10_v, _ccw_l10_l = _extract_combo_chart_window(recent_10,              combo_stats)
            _ccw_l20_v, _ccw_l20_l = _extract_combo_chart_window(_l20_combo,             combo_stats)
            _ccw_home_v, _ccw_home_l = _extract_combo_chart_window(home_games.head(10),  combo_stats)
            _ccw_away_v, _ccw_away_l = _extract_combo_chart_window(away_games.head(10),  combo_stats)
            _combo_chart_windows: dict = {
                "l5":   {"values": _ccw_l5_v,   "labels": _ccw_l5_l},
                "l10":  {"values": _ccw_l10_v,  "labels": _ccw_l10_l},
                "l20":  {"values": _ccw_l20_v,  "labels": _ccw_l20_l},
                "home": {"values": _ccw_home_v, "labels": _ccw_home_l},
                "away": {"values": _ccw_away_v, "labels": _ccw_away_l},
            }

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
                "l5_avg": round(float(l5_combo.mean()), 1) if len(l5_combo) > 0 else round(total_avg, 1),
                "model_pred": round(total_avg, 1),
                "stat_std": round(float(raw_combo.std()), 2) if len(raw_combo) > 1 else 3.0,
                "model_prob": None, "implied_prob": None, "edge": None,
                "has_live_odds": False,
                "live_line": None, "live_over_price": None,
                "live_under_price": None, "live_bookmaker": None,
                "hit_rate_l5": hit_rate_l5_combo,
                "hit_rate_vs_book": None,
                "hits_vs_book": None,
                "l5_values": [round(float(v), 1) for v in l5_combo.tolist()],
                "chart_windows": _combo_chart_windows,
            })

        # ── Double-Double and Triple-Double detection ─────────────────────────
        # DD: player scored 10+ in 2 of (PTS, REB, AST) in the same game
        # TD: player scored 10+ in all 3 of (PTS, REB, AST) in the same game
        for dd_label, dd_req in [("DD", 2), ("TD", 3)]:
            dd_stats = ["PTS", "REB", "AST"]
            if not all(s in recent_10.columns for s in dd_stats):
                continue
            _dd_data = recent_10[dd_stats].apply(pd.to_numeric, errors="coerce")
            # Count how many of (PTS, REB, AST) were >= 10 in each game
            _dd_hits_raw = (_dd_data >= 10).sum(axis=1)
            # Game "hits" = games where player had dd_req or more stats >= 10
            _dd_game_hits = (_dd_hits_raw >= dd_req).sum()
            _dd_total = len(_dd_hits_raw.dropna())
            if _dd_total < 5:
                continue
            _dd_hit_rate = _dd_game_hits / _dd_total
            if _dd_hit_rate < 0.40:  # only show if they hit DD/TD 40%+ of the time
                continue

            _dd_avg_pts = float(_dd_data["PTS"].mean() or 0)
            _dd_avg_reb = float(_dd_data["REB"].mean() or 0)
            _dd_avg_ast = float(_dd_data["AST"].mean() or 0)
            _dd_ev = calculate_ev(_dd_hit_rate)
            if blowout_risk and role in ("star", "starter"):
                _dd_ev *= 0.85
            _dd_ev *= consistency_mult

            props_data.append({
                "player":       player_name,
                "team":         player_team,
                "opponent":     opponent,
                "position":     position,
                "role":         role,
                "stat":         dd_label,
                "stat_label":   "DOUBLE-DOUBLE" if dd_label == "DD" else "TRIPLE-DOUBLE",
                "line":         1.5,   # conceptually "1+ DD" — must occur
                "avg":          round(_dd_hit_rate * 100, 1),
                "l5_avg":       round(float((_dd_hits_raw >= dd_req).head(5).mean()) * 100, 1),
                "model_pred":   round(_dd_hit_rate * 100, 1),
                "stat_std":     0.0,
                "direction":    "Over",
                "hit_rate":     _dd_hit_rate,
                "hits":         int(_dd_game_hits),
                "total":        _dd_total,
                "def_rank":     None,
                "is_home_today": is_home_today,
                "ev":           _dd_ev,
                "is_lock":      _dd_hit_rate >= 0.75 and _dd_total >= 5,
                "is_combo":     True,
                "blowout_risk":   blowout_risk,
                "blowout_spread": blowout_spread,
                "game_matchup": game_matchup_str,
                "hit_rate_home": 0.0, "hit_rate_away": 0.0,
                "hits_home": 0, "hits_away": 0,
                "total_home": 0, "total_away": 0,
                "avg_home": 0.0, "avg_away": 0.0,
                "insight": {
                    "narrative": f"{player_name} has {dd_label} in {int(_dd_game_hits)}/{_dd_total} recent games ({_dd_hit_rate*100:.0f}%). Avg: {_dd_avg_pts:.1f}pts / {_dd_avg_reb:.1f}reb / {_dd_avg_ast:.1f}ast."
                },
                "value_score": _dd_hit_rate,
                "injury_boost": _injury_boost_note,
                "model_prob": None, "implied_prob": None, "edge": None,
                "has_live_odds": False,
                "live_line": None, "live_over_price": None,
                "live_under_price": None, "live_bookmaker": None,
                "book_line": None,
                # L5 hit rate — used as primary quality gate signal
                "hit_rate_l5": round(
                    float((_dd_hits_raw >= dd_req).head(5).sum()) / min(5, len(_dd_hits_raw))
                    if len(_dd_hits_raw) > 0 else 0.0, 4
                ),
                "hit_rate_vs_book": None,
                "hits_vs_book": None,
                "l5_values": [round(float(v), 1) for v in (_dd_hits_raw >= dd_req).head(5).tolist()],
                "chart_windows": {},
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
    #   - Store sportsbook line as book_line (do NOT overwrite our value line)
    #   - Keep hit_rate against our value line (computed in _make_prop)
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
            prop["book_line"]        = sb_line      # store book line separately
            # DO NOT overwrite prop["line"] — keep our value line

            # Compute hit_rate against the actual book line (for confidence display)
            _l5v = prop.get("l5_values", [])
            if _l5v and sb_line:
                _dir = prop.get("direction", "Over")
                _hvb = sum(1 for v in _l5v if (v >= sb_line if _dir == "Over" else v < sb_line))
                prop["hit_rate_vs_book"] = round(_hvb / len(_l5v), 4)
                prop["hits_vs_book"]     = _hvb

            # DO NOT recalculate hit_rate against sb_line — keep hit_rate at value line
            # (hit_rate was computed against our value line during _make_prop and is correct)

            # EV = model's probability estimate vs sportsbook's implied probability
            # Use sb_line as the anchor for probability — we're asking: "what's the prob
            # our projection exceeds the BOOK line?" (used for EV calculation only)
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

            prop["model_prob"]      = round(model_prob, 4)
            prop["implied_prob"]    = round(implied, 4)
            prop["edge"]            = round(model_prob - implied, 4)
            prop["ev"]              = calculate_ev(model_prob, over_american=sb_price)
            prop["model_over_odds"] = _prob_to_american(model_prob)
            prop["model_under_odds"] = _prob_to_american(1.0 - model_prob)

        else:
            # No live odds: use model_prob vs default -110 implied (52.4%)
            model_prob = calculate_hit_probability(
                prediction=float(_mpred),
                line=float(prop["line"]),
                std_dev=_std,
                direction=prop.get("direction", "Over").lower(),
                stat_type=stat,
            )
            prop["model_prob"]       = round(model_prob, 4)
            prop["implied_prob"]     = 0.524
            prop["edge"]             = round(model_prob - 0.524, 4)
            prop["ev"]               = calculate_ev(model_prob)
            prop["model_over_odds"]  = _prob_to_american(model_prob)
            prop["model_under_odds"] = _prob_to_american(1.0 - model_prob)
            prop["has_live_odds"]    = False
            prop["live_line"]        = None
            prop["live_over_price"]  = None
            prop["live_under_price"] = None
            prop["live_bookmaker"]   = None
            # Use simulated book line as book_line reference when no live odds
            _sim_bl = prop.get("sim_book_line") or None
            prop["book_line"] = _sim_bl
            # Compute hit_rate against sim book line for confidence display
            _l5v = prop.get("l5_values", [])
            if _l5v and _sim_bl:
                _dir = prop.get("direction", "Over")
                _hvb = sum(1 for v in _l5v if (v >= _sim_bl if _dir == "Over" else v < _sim_bl))
                prop["hit_rate_vs_book"] = round(_hvb / len(_l5v), 4)
                prop["hits_vs_book"]     = _hvb

    # ── Post-enrichment: recompute hit_rate_l5 against the display line ──────
    # After enrichment, book_line is set (live or sim). Recompute hit_rate_l5
    # against whichever line will be shown in the chart so confidence % matches.
    for _prop in props_data:
        _disp = _prop.get("book_line") or _prop.get("sim_book_line") or _prop.get("line")
        _l5v  = _prop.get("l5_values") or []
        if _l5v and _disp is not None:
            _dir = _prop.get("direction", "Over")
            _h   = sum(1 for v in _l5v if (v >= float(_disp) if _dir == "Over" else v < float(_disp)))
            _prop["hit_rate_l5"] = round(_h / len(_l5v), 4)

    # ── Quality gate ─────────────────────────────────────────────────────────
    # Primary signal: L5 hit rate (most recent 5 games vs the specific line).
    # If a player hit 3+ of their last 5, we show it regardless of model_prob.
    # Secondary signals: model_prob + EV catch props with fewer L5 data points.
    _MIN_MODEL_PROB    = 0.53
    _MIN_EV            = 0.04
    _MIN_ODDS_AMERICAN = -180
    # Higher caps so each stat category can surface 15+ unique props
    _MAX_PER_PLAYER_TIER = {"star": 8, "starter": 6, "rotation": 5, "bench": 3}
    _HARD_CAP          = 300   # raise hard cap to accommodate 15+ per stat type
    _MEANINGFUL_LINE_FLOORS = {"PTS": 9.5, "AST": 3.5, "REB": 4.5}

    todays_stars: set[str] = {p["player"] for p in props_data if p.get("role") in ("star", "starter", "rotation")}

    quality_props = []
    for p in props_data:
        stat = p.get("stat", "")
        _raw_l5hr = p.get("hit_rate_l5")
        l5hr = _raw_l5hr if _raw_l5hr is not None else 0.0
        mp   = p.get("model_prob") or 0.0
        ev   = p.get("ev") or 0.0
        odd  = p.get("model_over_odds") or -300

        # Star/Starter/Rotation always-show — evaluated FIRST so the floor check
        # never silently drops a key player with a low line (e.g. rotation guard
        # with PTS line 8.5 or AST line 2.5).
        if p.get("player") in todays_stars:
            quality_props.append(p)
            continue

        # Floor check: bench/unknown players only.  Prevents trivial prop lines
        # like "Over 1.5 AST" from a deep bench player with no real edge.
        floor_val = _MEANINGFUL_LINE_FLOORS.get(stat)
        if floor_val and p.get("line", 0) < floor_val:
            continue

        # Primary gate: L5 hit rate ≥ 60% (3/5 games) — no model_prob required
        if l5hr >= 0.60:
            quality_props.append(p)
            continue
        # Combo gate — combos have no model_over_odds so skip the odds check
        if p.get("is_combo") and p.get("hit_rate", 0) >= 0.60:
            quality_props.append(p)
            continue
        # Standard model gate (for props with sparse L5 data)
        if mp >= _MIN_MODEL_PROB and ev >= _MIN_EV and odd >= _MIN_ODDS_AMERICAN:
            quality_props.append(p)
            continue
        # High book hit rate override
        hvb = p.get("hit_rate_vs_book") or 0.0
        if hvb >= 0.80 and odd >= -200:
            quality_props.append(p)

    # Deduplicate: tiered cap per player, BUT combos are never blocked by the cap.
    # Rule: individual-stat props compete for the per-role cap; combo/DD/TD props
    # always pass through (max 1 per (player, stat) pair to avoid dups).
    from collections import defaultdict as _dd
    _per_player: dict = _dd(list)
    for p in quality_props:
        _per_player[p["player"]].append(p)

    _COMBO_STATS = {"PTS+REB", "PTS+AST", "AST+REB", "PTS+AST+REB", "DD", "TD"}

    deduped: list = []
    for _pprops in _per_player.values():
        _pprops.sort(key=lambda x: (
            -(x.get("hit_rate_l5") or x.get("hit_rate_vs_book") or x.get("hit_rate") or 0),
            -(x.get("ev") or 0),
        ))
        role_key = _pprops[0].get("role", "bench")
        cap = _MAX_PER_PLAYER_TIER.get(role_key, 3)

        individual = [p for p in _pprops if p.get("stat", "") not in _COMBO_STATS]
        combos     = [p for p in _pprops if p.get("stat", "") in _COMBO_STATS]

        # Deduplicate combos: 1 per stat type per player
        seen_combo_stat: set = set()
        unique_combos: list  = []
        for p in combos:
            if p["stat"] not in seen_combo_stat:
                seen_combo_stat.add(p["stat"])
                unique_combos.append(p)

        deduped.extend(individual[:cap])
        deduped.extend(unique_combos)  # all combos always included

    # Final sort: L5 hit rate first (explicit None check), then book hit rate, then EV
    def _final_sort_key(x):
        l5 = x.get("hit_rate_l5")
        primary = l5 if l5 is not None else (x.get("hit_rate_vs_book") or x.get("hit_rate") or 0)
        return (-primary, -(x.get("ev") or 0), -(x.get("value_score") or 0))
    deduped.sort(key=_final_sort_key)
    props_data = deduped[:_HARD_CAP]

    print(f"[PropsCache] Quality gate: {len(props_data)} props (L5 hit rate primary gate, per-player caps raised)")

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

    # ── Build game predictions from spreads for ML parlay ─────────────────
    # win_prob ≈ 50% + |spread| / 28  (Massey-Peabody approximation)
    # Confidence thresholds: HIGH ≥ 68%, MEDIUM ≥ 58%, LOW < 58%
    game_predictions_for_parlay: list[dict] = []
    for home_team, opponent in game_info.get("team_to_opponent", {}).items():
        is_home = game_info.get("teams_home_away", {}).get(home_team, "home") == "home"
        if not is_home:
            continue  # only process each game once (home team entry)
        away_team = opponent
        spread    = game_spreads.get(home_team)   # negative = home favored
        if spread is None:
            continue
        home_win_prob = min(0.85, max(0.40, 0.50 - spread / 28.0))  # spread is home_line (neg = home favored)
        if home_win_prob >= 0.68:
            winner, conf, wp = home_team, "HIGH",   home_win_prob
        elif home_win_prob >= 0.58:
            winner, conf, wp = home_team, "MEDIUM", home_win_prob
        elif (1 - home_win_prob) >= 0.68:
            winner, conf, wp = away_team, "HIGH",   1 - home_win_prob
        elif (1 - home_win_prob) >= 0.58:
            winner, conf, wp = away_team, "MEDIUM", 1 - home_win_prob
        else:
            winner, conf, wp = home_team, "LOW",    home_win_prob
        game_predictions_for_parlay.append({
            "home": home_team, "away": away_team,
            "winner_pick": winner, "winner_confidence": conf,
            "spread": spread,        # home line (negative = home favored)
            "model_total": None,     # not yet computed — totals parlay skipped gracefully
        })

    # ── Build all parlays ──────────────────────────────────────────────────
    try:
        from utils.parlay_builder import build_all_parlays
        parlays_data = build_all_parlays(
            props=main_data,
            alt_lines=alt_lines,
            game_predictions=game_predictions_for_parlay,
        )
        print(f"[PropsCache] Parlays built: {parlays_data.get('total_count', 0)} total parlays")
        try:
            from utils.parlay_tracker import save_daily_parlays
            save_daily_parlays(today_str, parlays_data)
        except Exception as _pte:
            print(f"[PropsCache] Parlay save error (non-fatal): {_pte}")
    except Exception as _pe:
        print(f"[PropsCache] Parlay build error (non-fatal): {_pe}")
        parlays_data = {"over": [], "ml": [], "spread": [], "totals": [], "alt_over": [], "reduced": [], "alt": [], "under": [], "defense": [], "total_count": 0}

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
