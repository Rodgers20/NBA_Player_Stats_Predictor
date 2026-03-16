# utils/game_predictor.py
"""
Game-Level Spread & Total Predictor
=====================================
Uses team offensive/defensive ratings + recent form from player game logs
to predict game totals and spreads — no random guessing.

Methodology:
  predicted_home_score = blend(home_off_ppg, away_def_ppg) + home_court_adj
  predicted_away_score = blend(away_off_ppg, home_def_ppg)
  predicted_total      = predicted_home + predicted_away
  predicted_spread     = predicted_home - predicted_away  (neg = home favored)

  COVER PICK → compare model spread to actual spread
  TOTAL PICK → compare model total to actual O/U line
"""

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
HOME_COURT_ADV   = 2.5    # points home teams score extra on average
LEAGUE_AVG_PACE  = 98.0   # league-average possessions per 48 min
OFFENSE_WEIGHT   = 0.55   # blend weight for team's own offense vs opp defense
DEFENSE_WEIGHT   = 0.45
MIN_EDGE_SPREAD  = 1.5    # min model vs line gap to make a spread pick
MIN_EDGE_TOTAL   = 2.0    # min model vs line gap to make a total pick
RECENT_N_GAMES   = 10     # recent form window


class GamePredictor:
    """
    Predicts game totals and spreads using:
    - Season-level team defensive stats (OPP_PTS, W_PCT, PLUS_MINUS)
    - Rolling team offensive PPG derived from player game logs
    - Recent W/L form from player game logs
    - Home court advantage + pace adjustment
    """

    def __init__(self, team_def_df: pd.DataFrame, player_logs_df: pd.DataFrame):
        """
        Args:
            team_def_df:    team_defensive_stats.csv (one row per team per season)
            player_logs_df: engineered player game log data (DF global in app.py)
        """
        self.team_def  = team_def_df
        self.logs      = player_logs_df

        # Pre-build team game score table from player logs
        self._team_game_scores = self._build_team_game_scores()

    # ── Public API ────────────────────────────────────────────────────────

    def predict_game(self, home_abbr: str, away_abbr: str) -> dict:
        """
        Predict score, spread, and total for a matchup.

        Returns:
            {
              "predicted_home_score": float,
              "predicted_away_score": float,
              "predicted_total":      float,
              "predicted_spread":     float,   # home - away (negative = home favored)
              "home_form":            dict,
              "away_form":            dict,
              "home_season":          dict,
              "away_season":          dict,
              "confidence":           str,      # "HIGH" | "MEDIUM" | "LOW"
              "intel":                list[str],
            }
        """
        home_season = self._get_team_season_stats(home_abbr)
        away_season = self._get_team_season_stats(away_abbr)
        home_form   = self._get_team_form(home_abbr)
        away_form   = self._get_team_form(away_abbr)

        # Offensive PPG — prefer recent rolling avg, fall back to season derivation
        home_off = home_form.get("rolling_ppg") or home_season.get("ppg", 112.0)
        away_off = away_form.get("rolling_ppg") or away_season.get("ppg", 112.0)

        # Defensive PPG allowed (from team_def season stats)
        home_def_allowed = home_season.get("opp_ppg", 112.0)
        away_def_allowed = away_season.get("opp_ppg", 112.0)

        # Blend offense vs opponent defense
        raw_home = (home_off * OFFENSE_WEIGHT) + (away_def_allowed * DEFENSE_WEIGHT)
        raw_away = (away_off * OFFENSE_WEIGHT) + (home_def_allowed * DEFENSE_WEIGHT)

        # Apply home court advantage
        raw_home += HOME_COURT_ADV

        # Pace adjustment: scale by avg pace relative to league average
        home_pace = home_season.get("pace", LEAGUE_AVG_PACE)
        away_pace = away_season.get("pace", LEAGUE_AVG_PACE)
        avg_pace  = (home_pace + away_pace) / 2
        pace_factor = avg_pace / LEAGUE_AVG_PACE

        pred_home  = round(raw_home * pace_factor, 1)
        pred_away  = round(raw_away * pace_factor, 1)
        pred_total = round(pred_home + pred_away,  1)
        pred_spread = round(pred_home - pred_away, 1)

        # Confidence: based on how complete the data is
        data_quality = sum([
            bool(home_form.get("rolling_ppg")),
            bool(away_form.get("rolling_ppg")),
            not self.team_def.empty,
        ])
        confidence = ["LOW", "MEDIUM", "MEDIUM", "HIGH"][data_quality]

        intel = self._build_intel(
            home_abbr, away_abbr,
            home_season, away_season,
            home_form, away_form,
            pred_home, pred_away, pred_total, pred_spread,
        )

        return {
            "predicted_home_score": pred_home,
            "predicted_away_score": pred_away,
            "predicted_total":      pred_total,
            "predicted_spread":     pred_spread,
            "home_form":            home_form,
            "away_form":            away_form,
            "home_season":          home_season,
            "away_season":          away_season,
            "confidence":           confidence,
            "intel":                intel,
        }

    def get_pick(self, prediction: dict, actual_spread: float | None,
                 actual_total: float | None) -> dict:
        """
        Compare model prediction to sportsbook lines and return picks.

        Returns:
            {
              "spread_pick":       str | None,   e.g. "HOME -6.5"
              "spread_team":       str | None,   e.g. "BOS"
              "spread_confidence": str,
              "total_pick":        str | None,   e.g. "UNDER 224.5"
              "total_confidence":  str,
              "model_spread":      float,
              "model_total":       float,
            }
        """
        model_spread = prediction["predicted_spread"]
        model_total  = prediction["predicted_total"]
        result = {
            "spread_pick":       None,
            "spread_team":       None,
            "spread_confidence": "LOW",
            "total_pick":        None,
            "total_confidence":  "LOW",
            "model_spread":      model_spread,
            "model_total":       model_total,
        }

        # ── Spread pick ────────────────────────────────────────────────
        if actual_spread is not None:
            # actual_spread = the line for the home team (e.g. -6.5 means home favored 6.5)
            # model_spread  = home - away (e.g. -4.3 = home favored by 4.3)
            edge = model_spread - actual_spread  # how much model vs line
            if edge > MIN_EDGE_SPREAD:
                # Model thinks home wins by MORE than the line → HOME covers
                result["spread_pick"] = "HOME"
                result["spread_team"] = "home"
            elif edge < -MIN_EDGE_SPREAD:
                # Model thinks home wins by LESS than line → AWAY covers
                result["spread_pick"] = "AWAY"
                result["spread_team"] = "away"
            abs_edge = abs(edge)
            result["spread_confidence"] = (
                "HIGH" if abs_edge >= 4.0 else "MEDIUM" if abs_edge >= 2.0 else "LOW"
            )

        # ── Total pick ─────────────────────────────────────────────────
        if actual_total is not None:
            edge = model_total - actual_total
            if edge > MIN_EDGE_TOTAL:
                result["total_pick"] = "OVER"
            elif edge < -MIN_EDGE_TOTAL:
                result["total_pick"] = "UNDER"
            abs_edge = abs(edge)
            result["total_confidence"] = (
                "HIGH" if abs_edge >= 5.0 else "MEDIUM" if abs_edge >= 3.0 else "LOW"
            )

        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_team_game_scores(self) -> pd.DataFrame:
        """
        Aggregate player logs into team-level per-game PTS.
        MATCHUP[:3] = player's team abbreviation (e.g. 'BOS vs. MIA' → 'BOS').
        """
        if self.logs.empty or "MATCHUP" not in self.logs.columns:
            return pd.DataFrame()

        df = self.logs.copy()
        df["team_abbr"] = df["MATCHUP"].str[:3].str.strip()

        # Sum all players' PTS for (team, date) = team total points
        agg = (
            df.groupby(["team_abbr", "_date"])
            .agg(team_pts=("PTS", "sum"), wl=("WL", "first"))
            .reset_index()
            .sort_values("_date", ascending=False)
        )
        return agg

    def _get_team_season_stats(self, team_abbr: str) -> dict:
        """
        Pull the most recent season row for a team from team_def_df.
        Derives offensive PPG as OPP_PTS + PLUS_MINUS (point differential).
        """
        if self.team_def.empty or "TEAM_ABBREVIATION" not in self.team_def.columns:
            return {"ppg": 112.0, "opp_ppg": 112.0, "w_pct": 0.5, "pace": LEAGUE_AVG_PACE}

        rows = self.team_def[self.team_def["TEAM_ABBREVIATION"] == team_abbr]
        if rows.empty:
            return {"ppg": 112.0, "opp_ppg": 112.0, "w_pct": 0.5, "pace": LEAGUE_AVG_PACE}

        # Use the most recent season row
        row = rows.sort_values("SEASON", ascending=False).iloc[0]
        opp_pts    = float(row.get("OPP_PTS",    112.0))
        plus_minus = float(row.get("PLUS_MINUS",   0.0))
        w_pct      = float(row.get("W_PCT",         0.5))
        pace       = float(row["PACE"]) if "PACE" in row.index else LEAGUE_AVG_PACE
        opp_pts_rank = int(row.get("OPP_PTS_RANK", 15))

        return {
            "ppg":          round(opp_pts + plus_minus, 1),
            "opp_ppg":      round(opp_pts, 1),
            "w_pct":        round(w_pct, 3),
            "pace":         round(pace, 1),
            "opp_pts_rank": opp_pts_rank,
        }

    def _get_team_form(self, team_abbr: str, n: int = RECENT_N_GAMES) -> dict:
        """
        Compute rolling form for a team from aggregated game scores.
        """
        if self._team_game_scores.empty:
            return {"wins": 0, "losses": 0, "rolling_ppg": None, "wl_list": []}

        team_games = (
            self._team_game_scores[self._team_game_scores["team_abbr"] == team_abbr]
            .sort_values("_date", ascending=False)
            .head(n)
        )

        if team_games.empty:
            return {"wins": 0, "losses": 0, "rolling_ppg": None, "wl_list": []}

        wins   = int((team_games["wl"] == "W").sum())
        losses = int((team_games["wl"] == "L").sum())
        rolling_ppg = round(float(team_games["team_pts"].mean()), 1) if len(team_games) >= 3 else None
        wl_list = team_games["wl"].tolist()

        return {
            "wins":        wins,
            "losses":      losses,
            "rolling_ppg": rolling_ppg,
            "wl_list":     wl_list,
            "games":       len(team_games),
        }

    def _build_intel(self, home: str, away: str,
                     home_s: dict, away_s: dict,
                     home_f: dict, away_f: dict,
                     ph: float, pa: float, pt: float, ps: float) -> list:
        """Generate 2-4 human-readable insight bullets."""
        bullets = []

        # Defense quality
        home_def_rank = home_s.get("opp_pts_rank", 15)
        away_def_rank = away_s.get("opp_pts_rank", 15)
        if home_def_rank <= 8:
            bullets.append(f"{home} ranks #{home_def_rank} defense ({home_s['opp_ppg']:.0f} OPP PPG)")
        elif away_def_rank <= 8:
            bullets.append(f"{away} ranks #{away_def_rank} defense ({away_s['opp_ppg']:.0f} OPP PPG)")

        # Recent form note
        hw, hl = home_f.get("wins", 0), home_f.get("losses", 0)
        aw, al = away_f.get("wins", 0), away_f.get("losses", 0)
        if hw + hl >= 5:
            bullets.append(f"{home} ({hw}-{hl} last {hw+hl}) | {away} ({aw}-{al} last {aw+al})")

        # Rolling PPG comparison
        h_ppg = home_f.get("rolling_ppg")
        a_ppg = away_f.get("rolling_ppg")
        if h_ppg and a_ppg:
            bullets.append(f"Rolling PPG — {home}: {h_ppg:.0f} | {away}: {a_ppg:.0f}")

        # Model total vs typical
        if pt < 215:
            bullets.append(f"Model projects low-scoring game ({pt:.0f} total) — defensive matchup")
        elif pt > 230:
            bullets.append(f"Model projects high-scoring game ({pt:.0f} total) — fast-paced matchup")

        return bullets[:4]
