# utils/game_predictor.py
"""
Game-Level Predictor — Spread, Total, and Winner
==================================================
Uses team recent form (rolling 10 games) + season defensive ratings
+ home court advantage + pace + head-to-head history.

Loads a validated league artifact when available, otherwise uses an explicitly
unvalidated historical-form fallback. All inputs are cut off before game day.
"""

import logging
import math
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
HOME_COURT_ADV   = 2.5    # pts home teams score extra on average
LEAGUE_AVG_PACE  = 98.0
FORM_WEIGHT      = 0.65   # recent rolling form vs season rating
SEASON_WEIGHT    = 0.35
MIN_EDGE_SPREAD  = 1.5    # min gap to issue a spread pick
MIN_EDGE_TOTAL   = 2.0    # min gap to issue a total pick
RECENT_N         = 10     # rolling form window
H2H_SEASONS      = 2      # how many seasons back for H2H
FORM_WP_SCALE    = 10.0   # multiplier for recent W/L rate vs season rate
MARGIN_SCALE     = 0.30   # weight for rolling scoring-margin adjustment

# ── Rest / fatigue constants ───────────────────────────────────────────────
B2B_PENALTY  = 2.5   # pts deducted for back-to-back (rest_days == 0)
REST_BONUS   = 0.8   # small bonus when team has 3+ days rest

# Injury penalty — replacement-level model
# NBA bench/replacement player averages ~7 PPG.
# Only the MARGINAL gap above replacement is penalized, and only partially
# because the team redistributes usage (other starters get more minutes,
# someone else steps up). A 20 PPG star being out ≠ team loses 14 pts.
# Weights increased (was 0.20/0.10) to better reflect real multi-player impact.
REPLACEMENT_PPG        = 7.0   # typical bench/replacement scorer
OUT_MARGINAL_WEIGHT    = 0.40  # 40% of net marginal loss → team scoring drop
DOUBTFUL_MARGINAL_WEIGHT = 0.20
MAX_PENALTY_PER_PLAYER = 6.0   # hard cap per player (pts)
MAX_TOTAL_PENALTY      = 14.0  # hard cap total per team (pts)


class GamePredictor:
    """
    Predicts winner, spread, and total using:
      - Rolling 10-game team PPG (from player game logs)
      - Team defensive ratings (OPP_PTS from team_def_df)
      - Recent W/L form momentum
      - Head-to-head record
      - Home court advantage + pace adjustment
      - Rest/fatigue (back-to-back penalty, 3+ days rest bonus)
      - Validated league-specific margin/total artifact when promoted
    """

    def __init__(self, team_def_df: pd.DataFrame, player_logs_df: pd.DataFrame,
                 league: str = "nba", target_date=None):
        if league not in ("nba", "wnba"):
            raise ValueError("Unsupported basketball league")
        self.league = league
        self.target_date = target_date
        self._all_logs = player_logs_df.copy()
        self._team_history = None
        try:
            from utils.game_data import load_game_team_history
            self._team_history = load_game_team_history(league)
        except (ImportError, FileNotFoundError):
            pass
        except Exception:
            logger.exception("Team-game history invalid; retaining explicit player-log fallback")
        self.team_def = team_def_df
        self.logs     = player_logs_df
        self._scores  = self._build_scores(player_logs_df)

    def refresh(self, player_logs_df: pd.DataFrame) -> None:
        """Re-build the game scores table from a fresh DF (call after data update)."""
        self._all_logs = player_logs_df.copy()
        self.logs    = player_logs_df
        self._scores = self._build_scores(player_logs_df)

    # ── Public API ────────────────────────────────────────────────────────

    def predict_game(self, home: str, away: str,
                     home_injuries: list | None = None,
                     away_injuries: list | None = None,
                     target_date=None, neutral: bool = False, venue=None) -> dict:
        """
        Full game prediction, optionally adjusted for known injuries.

        Args:
            home: home team abbreviation
            away: away team abbreviation
            home_injuries: list of {name, status, reason} dicts for home team
            away_injuries: list of {name, status, reason} dicts for away team

        Returns
        -------
        {
          predicted_home_score, predicted_away_score, predicted_total,
          predicted_spread, winner, winner_margin, winner_confidence,
          home_form, away_form, h2h, home_season, away_season,
          intel, home_missing, away_missing,
          reasoning: {winner_reason, total_reason, spread_reason, key_factors}
        }
        """
        self._set_cutoff(target_date or self.target_date)
        home_form   = self._get_form(home)
        away_form   = self._get_form(away)
        home_season = self._get_season(home)
        away_season = self._get_season(away)
        h2h         = self.get_h2h(home, away)

        # ── Offensive estimate per team ───────────────────────────────────
        home_off_raw = home_form.get("rolling_ppg") or home_season.get("ppg", 112.0)
        away_off_raw = away_form.get("rolling_ppg") or away_season.get("ppg", 112.0)

        # Prefer rolling defensive PPG over season average — captures recent hot/cold
        # streaks like a 0-10 team allowing 130 PPG in last 10 games
        home_def_raw = home_form.get("rolling_opp_ppg") or home_season.get("opp_ppg", 112.0)
        away_def_raw = away_form.get("rolling_opp_ppg") or away_season.get("opp_ppg", 112.0)

        # Blend form offense vs opponent's recent defense
        pred_home = (home_off_raw * FORM_WEIGHT + away_def_raw * SEASON_WEIGHT)
        pred_away = (away_off_raw * FORM_WEIGHT + home_def_raw * SEASON_WEIGHT)

        # Form momentum (W/L rate vs season baseline) — scaled by FORM_WP_SCALE
        home_form_wp   = home_form.get("wins", 0) / max(home_form.get("wins", 0) + home_form.get("losses", 0), 1) if home_form.get("games") else home_season.get("w_pct", 0.5)
        away_form_wp   = away_form.get("wins", 0) / max(away_form.get("wins", 0) + away_form.get("losses", 0), 1) if away_form.get("games") else away_season.get("w_pct", 0.5)
        home_season_wp = home_season.get("w_pct", 0.5)
        away_season_wp = away_season.get("w_pct", 0.5)
        pred_home += (home_form_wp - home_season_wp) * FORM_WP_SCALE
        pred_away += (away_form_wp - away_season_wp) * FORM_WP_SCALE

        # Rolling scoring-margin adjustment — teams that are getting crushed
        # even while scoring (e.g. 0-10 despite 118 PPG) get penalized here
        pred_home += home_form.get("rolling_margin", 0.0) * MARGIN_SCALE
        pred_away += away_form.get("rolling_margin", 0.0) * MARGIN_SCALE

        # H2H boost
        h2h_boost = self._h2h_boost(h2h, home, away)
        pred_home += h2h_boost
        pred_away -= h2h_boost

        # Home court advantage
        pred_home += 0.0 if neutral else (HOME_COURT_ADV if self.league == "nba" else 2.0)

        # Pace adjustment
        league_pace = 98.0 if self.league == "nba" else 80.0
        home_pace   = home_season.get("pace", league_pace)
        away_pace   = away_season.get("pace", league_pace)
        pace_factor = ((home_pace + away_pace) / 2) / league_pace
        pred_home   = round(pred_home * pace_factor, 1)
        pred_away   = round(pred_away * pace_factor, 1)

        # ── Rest / back-to-back fatigue ──────────────────────────────────
        home_rest = self._get_rest_days(home)
        away_rest = self._get_rest_days(away)
        if home_rest == 0:
            pred_home -= B2B_PENALTY
        elif home_rest >= 3:
            pred_home += REST_BONUS
        if away_rest == 0:
            pred_away -= B2B_PENALTY
        elif away_rest >= 3:
            pred_away += REST_BONUS

        forecast = None
        model_metadata = {"name": "historical_form_fallback", "validated": False,
                          "injury_adjustment_validated": False,
                          "market_assisted": False}
        try:
            from utils.game_features import aggregate_team_games, build_feature_vector
            from utils.game_model import GameForecastModel
            learned = GameForecastModel.load(self.league)
            if learned is not None:
                features = build_feature_vector(self._pregame_team_history() if self._team_history is not None else aggregate_team_games(self.logs, self.league),
                                                self._cutoff, home, away, self.league,
                                                neutral=neutral, venue=venue if neutral else None)
                candidate = learned.predict(features)
                trained_through = candidate.get("trained_through")
                if trained_through and pd.Timestamp(trained_through).normalize() >= self._cutoff:
                    raise ValueError("Artifact includes outcomes on or after the target date")
                forecast = candidate
                pred_home = (forecast["total"] + forecast["margin"]) / 2
                pred_away = (forecast["total"] - forecast["margin"]) / 2
                model_metadata.update(name=forecast["model_name"], validated=True,
                                      metrics=forecast.get("metrics", {}),
                                      version=forecast.get("version"),
                                      trained_through=forecast.get("trained_through"),
                                      feature_count=forecast.get("feature_count"),
                                      calibration=forecast.get("calibration"))
        except (ImportError, FileNotFoundError):
            pass
        except Exception:
            logger.exception("Validated game artifact unavailable; using historical fallback")
            model_metadata["artifact_unavailable"] = True

        # ── Injury adjustments ───────────────────────────────────────────
        home_missing, away_missing = [], []
        if home_injuries:
            pred_home, home_missing = self._apply_injury_penalty(pred_home, home_injuries, home)
            pred_away += sum(m.get("defensive_scenario_points", 0) for m in home_missing)
        if away_injuries:
            pred_away, away_missing = self._apply_injury_penalty(pred_away, away_injuries, away)
            pred_home += sum(m.get("defensive_scenario_points", 0) for m in away_missing)

        pred_total  = round(pred_home + pred_away, 1)
        pred_spread = round(pred_home - pred_away, 1)

        margin = abs(pred_spread)
        probability = None
        intervals = None
        if forecast is not None:
            if not home_missing and not away_missing:
                probability = forecast.get("home_win_probability")
            sd = float(forecast.get("margin_sd") or 0)
            if sd > 0 and (probability is None or home_missing or away_missing):
                probability = 0.5 * (1 + math.erf(pred_spread / (sd * math.sqrt(2))))
            empirical = forecast.get("prediction_intervals") or {}
            if empirical:
                margin_delta = pred_spread - forecast["margin"]
                total_delta = pred_total - forecast["total"]
                intervals = {"coverage": 0.80,
                             "margin": [round(v + margin_delta, 1) for v in empirical.get("margin_80", [])],
                             "total": [round(v + total_delta, 1) for v in empirical.get("total_80", [])],
                             "injury_scenario_shifted": bool(home_missing or away_missing)}
        winner_conf = "LOW"  # Compatibility label; no unsupported confidence claim.
        winner      = (home if probability >= 0.5 else away) if probability is not None else (home if pred_home >= pred_away else away)

        intel = self._build_intel(
            home, away, home_form, away_form,
            home_season, away_season, h2h,
            pred_home, pred_away, pred_total,
            home_missing, away_missing,
        )

        reasoning = self._build_reasoning(
            home, away, home_form, away_form,
            home_season, away_season, h2h,
            pred_home, pred_away, pred_total, pred_spread,
            winner, winner_conf, home_missing, away_missing,
        )

        model_metadata["injury_scenario_applied"] = bool(home_missing or away_missing)
        model_metadata["venue"] = venue
        model_metadata["venue_location_basis"] = "explicit_neutral_venue" if neutral else "home_arena_city_proxy"
        return {
            "model_metadata": model_metadata,
            "data_quality": self._data_quality(home, away, home_injuries, away_injuries),
            "home_win_probability": probability,
            "winner_probability": max(probability, 1-probability) if probability is not None else None,
            "prediction_intervals": intervals,
            "predicted_home_score": round(pred_home, 1),
            "predicted_away_score": round(pred_away, 1),
            "predicted_total":      pred_total,
            "predicted_spread":     pred_spread,
            "winner":               winner,
            "winner_margin":        margin,
            "winner_confidence":    winner_conf,
            "home_form":            home_form,
            "away_form":            away_form,
            "h2h":                  h2h,
            "home_season":          home_season,
            "away_season":          away_season,
            "intel":                intel,
            "home_missing":         home_missing,
            "away_missing":         away_missing,
            "reasoning":            reasoning,
        }

    def get_pick(self, prediction: dict, home: str, away: str,
                 actual_spread: float | None = None,
                 actual_total:  float | None = None) -> dict:
        """
        Compare model to sportsbook line and return picks.

        Returns
        -------
        {
          spread_pick, spread_team, spread_confidence,
          total_pick, total_confidence,
          winner_pick, winner_confidence,
          model_spread, model_total,
        }
        """
        model_spread = prediction["predicted_spread"]
        model_total  = prediction["predicted_total"]

        # Compare the displayed forecast directly with the quoted line.
        blended_spread, blended_total = model_spread, model_total

        result = {
            "spread_pick":        None,
            "spread_team":        None,
            "spread_confidence":  "LOW",
            "total_pick":         None,
            "total_confidence":   "LOW",
            "winner_pick":        prediction.get("winner"),
            "winner_confidence":  prediction.get("winner_confidence", "LOW"),
            "model_spread":       model_spread,
            "model_total":        model_total,
        }

        if actual_spread is not None:
            # actual_spread is the home line (negative = home favored, e.g. -5 means home -5).
            # Home covers when blended_spread > -actual_spread, i.e. blended_spread + actual_spread > 0.
            edge = blended_spread + actual_spread
            if edge > MIN_EDGE_SPREAD:
                result["spread_pick"] = "HOME"
                result["spread_team"] = home
            elif edge < -MIN_EDGE_SPREAD:
                result["spread_pick"] = "AWAY"
                result["spread_team"] = away
            abs_edge = abs(edge)
            result["spread_confidence"] = (
                "LOW"  # Edge size alone does not establish calibrated confidence.
            )

        if actual_total is not None:
            edge = blended_total - actual_total
            if edge > MIN_EDGE_TOTAL:
                result["total_pick"] = "OVER"
            elif edge < -MIN_EDGE_TOTAL:
                result["total_pick"] = "UNDER"
            abs_edge = abs(edge)
            result["total_confidence"] = (
                "LOW"  # Edge size alone does not establish calibrated confidence.
            )

        return result

    def get_team_last_n_games(self, team: str, n: int = 10) -> list[dict]:
        """
        Return the last n games for a team with date, opponent, W/L, and scores.

        Each dict:
          { date, opponent, wl, team_pts, opp_pts, home_away }
        """
        if self._scores.empty:
            return []

        team_games = (
            self._scores[self._scores["team"] == team]
            .sort_values("_date", ascending=False)
            .head(n)
        )

        results = []
        for _, row in team_games.iterrows():
            opp  = row.get("opponent", "")
            date = row["_date"]
            # look up opponent score on same date
            opp_game = self._scores[
                (self._scores["team"] == opp) & (self._scores["_date"] == date)
            ]
            opp_pts = round(float(opp_game["team_pts"].iloc[0]), 0) if not opp_game.empty else None

            results.append({
                "date":       date.strftime("%-m/%-d") if hasattr(date, "strftime") else str(date)[:10],
                "opponent":   opp,
                "wl":         row.get("wl", ""),
                "team_pts":   round(float(row["team_pts"]), 0),
                "opp_pts":    opp_pts,
                "home_away":  "HOME" if row.get("is_home") else "AWAY",
            })
        return results

    def get_h2h(self, team_a: str, team_b: str) -> dict:
        """
        Head-to-head record between two teams from available game log data.

        Returns
        -------
        {
          wins_a, losses_a, total,
          games: [ {date, wl, team_a_pts, team_b_pts}, ... ]   # last 5
        }
        """
        if self._scores.empty:
            return {"wins_a": 0, "losses_a": 0, "total": 0, "games": []}

        h2h_rows = (
            self._scores[
                (self._scores["team"] == team_a) &
                (self._scores["opponent"] == team_b)
            ]
            .sort_values("_date", ascending=False)
        )

        wins_a   = int((h2h_rows["wl"] == "W").sum())
        losses_a = int((h2h_rows["wl"] == "L").sum())

        games = []
        for _, row in h2h_rows.head(5).iterrows():
            date = row["_date"]
            b_game = self._scores[
                (self._scores["team"] == team_b) & (self._scores["_date"] == date)
            ]
            b_pts = round(float(b_game["team_pts"].iloc[0]), 0) if not b_game.empty else None
            a_pts = round(float(row["team_pts"]), 0)

            games.append({
                "date":       date.strftime("%-m/%-d") if hasattr(date, "strftime") else str(date)[:10],
                "wl":         row.get("wl", ""),
                "home_away":  "HOME" if row.get("is_home") else "AWAY",
                "team_a_pts": a_pts,
                "team_b_pts": b_pts,
            })

        return {
            "wins_a":   wins_a,
            "losses_a": losses_a,
            "total":    wins_a + losses_a,
            "games":    games,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _set_cutoff(self, target_date=None):
        self._cutoff = pd.Timestamp(target_date or datetime.now(ZoneInfo("America/New_York")).date()).normalize()
        logs = self._all_logs.copy()
        if "_date" not in logs:
            logs["_date"] = pd.to_datetime(logs.get("GAME_DATE"), errors="coerce")
        logs["_date"] = pd.to_datetime(logs["_date"], errors="coerce").dt.tz_localize(None)
        self.logs = logs[logs["_date"] < self._cutoff].copy()
        game_col = next((c for c in ("Game_ID", "GAME_ID") if c in self.logs), None)
        player_col = next((c for c in ("Player_ID", "PLAYER_ID", "PLAYER_NAME") if c in self.logs), None)
        if game_col and player_col:
            self.logs = self.logs.drop_duplicates([game_col, player_col], keep="last")
        self._scores = self._build_scores(self.logs)
        if self._team_history is not None:
            history = self._pregame_team_history()
            self._scores = history.rename(columns={"date": "_date", "home": "is_home", "PTS": "team_pts"}).copy()
            self._scores["wl"] = self._scores["margin"].map(lambda margin: "W" if margin > 0 else "L")
            self._scores = self._scores.sort_values("_date", ascending=False)

    def _pregame_team_history(self):
        return self._team_history[self._team_history["date"] < self._cutoff].copy()

    def _get_rest_days(self, team: str) -> int:
        rows = self._scores[self._scores["team"] == team] if not self._scores.empty else pd.DataFrame()
        if rows.empty:
            return 1
        cutoff = getattr(self, "_cutoff", pd.Timestamp(self.target_date or datetime.now(ZoneInfo("America/New_York")).date()))
        return min(max(int((cutoff - rows["_date"].max()).days) - 1, 0), 10)

    def _data_quality(self, home, away, home_injuries, away_injuries):
        dates = {}
        player_latest = self.logs["_date"].max() if not self.logs.empty else None
        for team in (home, away):
            rows = self._scores[self._scores["team"] == team] if not self._scores.empty else pd.DataFrame()
            last = rows["_date"].max() if not rows.empty else None
            player_rows = self.logs[self.logs["MATCHUP"].str.startswith(team + " ", na=False)] if "MATCHUP" in self.logs else pd.DataFrame()
            team_player_latest = player_rows["_date"].max() if not player_rows.empty else None
            dates[team] = {"latest_game_date": last.date().isoformat() if last is not None else None,
                           "history_age_days": int((self._cutoff-last).days) if last is not None else None,
                           "history_older_than_14_days": bool(last is None or (self._cutoff-last).days > 14),
                           "games_available": len(rows),
                           "latest_player_log_date": team_player_latest.date().isoformat() if team_player_latest is not None else None,
                           "player_boxscore_lag_days": max(int((last-team_player_latest).days), 0) if last is not None and team_player_latest is not None else None,
                           "injury_player_history_stale": bool(team_player_latest is None or (self._cutoff-team_player_latest).days > 7)}
        return {"as_of": self._cutoff.date().isoformat(), "teams": dates,
                "latest_player_log_date": player_latest.date().isoformat() if player_latest is not None else None,
                "player_history_age_days": int((self._cutoff-player_latest).days) if player_latest is not None else None,
                "injury_player_history_stale": bool(player_latest is None or (self._cutoff-player_latest).days > 7),
                "injury_feed_available": {home: home_injuries is not None, away: away_injuries is not None},
                "limitations": ["Injury scenario coefficients are not historically validated",
                                "No timestamped historical injury archive is used in training",
                                "Team box scores reconstructed from player logs"] if self._team_history is None else
                               ["Injury scenario coefficients are not historically validated",
                                "No timestamped historical injury archive is used in training"],
                "team_history_source": "validated_team_boxscores" if self._team_history is not None else "player_log_reconstruction"}

    def _apply_injury_penalty(self, raw_score: float, injuries: list, team: str | None = None) -> tuple:
        """
        Apply a realistic injury penalty using replacement-level logic.

        Key insight: when a player is OUT, a bench player or another starter
        absorbs their minutes. The team doesn't simply lose all of that scorer's
        production — someone else steps up. We model this as:

            net_marginal = player_ppg - REPLACEMENT_PPG  (what truly goes missing)
            penalty      = net_marginal * weight          (only a fraction of that)

        This means a 20 PPG star being OUT costs the team ~2-3 pts,
        not 14 pts. That's consistent with real-world data showing teams
        typically score ~3-5 fewer points when a star misses a game.

        Returns (adjusted_score, missing_list)
        missing_list items: {"name": str, "ppg": float, "status": str}
        """
        total_penalty = 0.0
        missing       = []

        for player in injuries:
            status = str(player.get("status", "")).strip().upper()
            if status not in ("OUT", "DOUBTFUL", "QUESTIONABLE"):
                continue
            name = player.get("name", "")

            # Look up player's L10 PPG from game logs
            normalize = lambda value: "".join(c for c in unicodedata.normalize("NFKD", str(value).casefold()) if c.isalnum())
            player_rows = self.logs[self.logs["PLAYER_NAME"].map(normalize) == normalize(name)]
            if team and "MATCHUP" in player_rows:
                player_rows = player_rows[player_rows["MATCHUP"].str.startswith(team + " ", na=False)]
            if len(player_rows) < 3:
                continue
            ppg = float(
                player_rows.sort_values("_date", ascending=False)
                .head(10)["PTS"]
                .mean()
            )

            recent = player_rows.sort_values("_date", ascending=False).head(10)
            team_dates = set(self._scores[self._scores["team"] == team].head(10)["_date"]) if team else set()
            participation = len(set(recent["_date"]) & team_dates) / max(len(team_dates), 1) if team_dates else 1.0
            status_weight = {"OUT": 1.0, "DOUBTFUL": 0.75, "QUESTIONABLE": 0.35}[status]
            average = lambda column: float(pd.to_numeric(recent[column], errors="coerce").fillna(0).mean()) if column in recent else 0.0
            minutes = average("MIN")
            net_marginal = max(ppg - (REPLACEMENT_PPG if self.league == "nba" else 5), 0.0)
            weight = OUT_MARGINAL_WEIGHT * status_weight * participation
            offense_proxy = net_marginal + 0.35*average("AST") + 0.08*average("REB")
            defensive_proxy = min(2.0, (0.25*average("STL") + 0.25*average("BLK") + 0.05*average("REB")) * min(minutes/30, 1.2)) * status_weight * participation
            player_penalty = min(offense_proxy * weight, MAX_PENALTY_PER_PLAYER)
            total_penalty += player_penalty
            missing.append({"name": name, "ppg": round(ppg, 1), "status": status,
                            "participation_fraction": round(participation, 3),
                            "offensive_scenario_points": round(player_penalty, 2),
                            "defensive_scenario_points": round(defensive_proxy, 2),
                            "adjustment_validated": False})

            if total_penalty >= MAX_TOTAL_PENALTY:
                break  # don't pile on more than the team cap

        total_penalty = min(total_penalty, MAX_TOTAL_PENALTY)
        adjusted = round(raw_score - total_penalty, 1)
        return adjusted, missing

    def _build_reasoning(self, home: str, away: str,
                          hf: dict, af: dict,
                          hs: dict, as_: dict,
                          h2h: dict,
                          ph: float, pa: float, pt: float, ps: float,
                          winner: str, winner_conf: str,
                          home_missing: list, away_missing: list) -> dict:
        """
        Build natural-language explanations for each pick.

        Returns dict with winner_reason, total_reason, spread_reason, key_factors.
        """
        loser   = away if winner == home else home
        margin  = abs(ps)

        # ── Winner reason ─────────────────────────────────────────────────
        hw, hl = hf.get("wins", 0), hf.get("losses", 0)
        aw, al = af.get("wins", 0), af.get("losses", 0)
        home_ppg_l10 = hf.get("rolling_ppg") or hs.get("ppg", 0)
        away_ppg_l10 = af.get("rolling_ppg") or as_.get("ppg", 0)

        winner_form = (hw, hl) if winner == home else (aw, al)
        winner_ppg  = home_ppg_l10 if winner == home else away_ppg_l10
        loser_form  = (aw, al) if winner == home else (hw, hl)
        loser_ppg   = away_ppg_l10 if winner == home else home_ppg_l10

        # Show winner's score first so "DET projected to win 99–87" reads correctly
        winner_score = ph if winner == home else pa
        loser_score  = pa if winner == home else ph
        winner_reason = (
            f"{winner} projected to win {winner_score:.0f}–{loser_score:.0f} "
            f"(margin {margin:.1f} pts, {winner_conf.lower()} conf). "
            f"{winner} L10: {winner_form[0]}-{winner_form[1]}, {winner_ppg:.0f} PPG avg. "
            f"{loser} L10: {loser_form[0]}-{loser_form[1]}, {loser_ppg:.0f} PPG avg."
        )

        # Injury note in winner reason
        missing_on_loser = away_missing if winner == home else home_missing
        for m in missing_on_loser[:2]:
            winner_reason += f" {loser} missing {m['name']} ({m['ppg']:.0f} PPG, {m['status']})."

        # ── Total reason ──────────────────────────────────────────────────
        pace_note = ""
        league_pace = 98.0 if self.league == "nba" else 80.0
        home_pace = hs.get("pace", league_pace)
        away_pace = as_.get("pace", league_pace)
        avg_pace  = (home_pace + away_pace) / 2
        if avg_pace > league_pace + 2:
            pace_note = f" Both teams play fast (avg pace {avg_pace:.0f} vs league {league_pace:.0f})."
        elif avg_pace < league_pace - 2:
            pace_note = f" Slow-pace matchup (avg {avg_pace:.0f})."

        home_def_l10 = hf.get("rolling_opp_ppg") or hs.get("opp_ppg", 0)
        away_def_l10 = af.get("rolling_opp_ppg") or as_.get("opp_ppg", 0)
        total_reason = (
            f"Model projects {pt:.0f} combined pts "
            f"({home} {ph:.0f} + {away} {pa:.0f})."
            f" {home} allows {home_def_l10:.0f} PPG L10, "
            f"{away} allows {away_def_l10:.0f} PPG L10.{pace_note}"
        )

        # ── Spread reason ─────────────────────────────────────────────────
        home_def_rank = hs.get("opp_pts_rank", 15)
        away_def_rank = as_.get("opp_pts_rank", 15)

        # Express spread from the FAVORED team's perspective:
        # ps = pred_home - pred_away. Negative → away team favored.
        # Convention: favorite gets the negative number (e.g. "DET -11.7").
        if ps >= 0:
            fav_team   = home
            fav_spread = f"-{ps:.1f}"      # home wins → "HOME -X.X"
        else:
            fav_team   = away
            fav_spread = f"{ps:.1f}"       # ps already negative → "AWAY -X.X"

        home_def_ppg = hf.get("rolling_opp_ppg") or hs.get("opp_ppg", 0)
        away_def_ppg = af.get("rolling_opp_ppg") or as_.get("opp_ppg", 0)

        spread_reason = (
            f"Model line: {fav_team} {fav_spread} (favored). "
            f"{home} defense #{home_def_rank} (allows {home_def_ppg:.0f} PPG L10), "
            f"{away} defense #{away_def_rank} (allows {away_def_ppg:.0f} PPG L10). "
            f"H2H: {home} {h2h.get('wins_a', 0)}-{h2h.get('losses_a', 0)} "
            f"in last {h2h.get('total', 0)} meetings."
        ) if h2h.get("total", 0) >= 1 else (
            f"Model line: {fav_team} {fav_spread} (favored). "
            f"{home} #{home_def_rank} defense, {away} #{away_def_rank} defense."
        )

        # ── Key factors (3-5 bullets for UI) ─────────────────────────────
        key_factors = []

        # Form
        if hw + hl >= 5 and aw + al >= 5:
            key_factors.append(
                f"Form L10: {home} {hw}-{hl} ({home_ppg_l10:.0f} PPG)  ·  "
                f"{away} {aw}-{al} ({away_ppg_l10:.0f} PPG)"
            )

        # Injuries on either side
        for m in (home_missing + away_missing)[:3]:
            team = home if m in home_missing else away
            key_factors.append(
                f"🚑 {m['name']} ({team}) — {m['status']} ({m['ppg']:.0f} PPG missing)"
            )

        # H2H note
        if h2h.get("total", 0) >= 3:
            key_factors.append(
                f"H2H last {h2h['total']} games: {home} {h2h['wins_a']}-{h2h['losses_a']}"
            )

        # Defense / pace
        if home_def_rank <= 5:
            key_factors.append(f"{home} elite D — #{home_def_rank} in pts allowed")
        elif away_def_rank <= 5:
            key_factors.append(f"{away} elite D — #{away_def_rank} in pts allowed")
        if pace_note:
            key_factors.append(pace_note.strip())

        # Winner context
        if margin >= 6:
            key_factors.append(f"{winner} projected to win by {margin:.0f} pts")

        return {
            "winner_reason": winner_reason,
            "total_reason":  total_reason,
            "spread_reason": spread_reason,
            "key_factors":   key_factors[:5],
        }

    def _build_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate player logs into team-level per-game scores.
        Extracts team + opponent from MATCHUP column.
        """
        if df is None or df.empty or "MATCHUP" not in df.columns:
            return pd.DataFrame()

        d = df.copy()
        game_col = next((c for c in ("Game_ID", "GAME_ID") if c in d), None)
        player_col = next((c for c in ("Player_ID", "PLAYER_ID", "PLAYER_NAME") if c in d), None)
        if game_col and player_col:
            d = d.drop_duplicates([game_col, player_col], keep="last")

        # Parse team and opponent from MATCHUP  e.g. "ATL vs. ORL" or "ATL @ ORL"
        d["team"]     = d["MATCHUP"].str.extract(r'^([A-Z]{2,3})\s+(?:vs\.|@)')[0]
        d["opponent"] = d["MATCHUP"].str.extract(r'(?:vs\.|@)\s+([A-Z]{2,3})')[0]
        d["is_home"]  = d["MATCHUP"].str.contains(r"\bvs\.", regex=True, na=False)

        # Drop rows with missing extraction or date
        d = d.dropna(subset=["team", "_date", "WL"])
        d["WL"] = d["WL"].astype(str).str.strip().str.upper()
        d = d[d["WL"].isin(["W", "L"])]

        # Sum player PTS per (team, date) = team total points per game
        agg = (
            d.groupby(["team", "_date", "opponent", "is_home"])
            .agg(
                team_pts=("PTS", "sum"),
                wl=("WL",  lambda x: "W" if (x == "W").sum() > (x == "L").sum() else "L"),
            )
            .reset_index()
            .sort_values("_date", ascending=False)
        )
        return agg

    def _get_form(self, team: str, n: int = RECENT_N) -> dict:
        """Rolling form for last n games, including defensive PPG and scoring margin."""
        if self._scores.empty:
            return {
                "wins": 0, "losses": 0, "rolling_ppg": None, "wl_list": [], "games": 0,
                "rolling_opp_ppg": None, "rolling_margin": 0.0,
            }

        rows = (
            self._scores[self._scores["team"] == team]
            .sort_values("_date", ascending=False)
            .head(n)
        )

        if rows.empty:
            return {
                "wins": 0, "losses": 0, "rolling_ppg": None, "wl_list": [], "games": 0,
                "rolling_opp_ppg": None, "rolling_margin": 0.0,
            }

        wins   = int((rows["wl"] == "W").sum())
        losses = int((rows["wl"] == "L").sum())
        rolling_ppg = round(float(rows["team_pts"].mean()), 1) if len(rows) >= 3 else None
        wl_list = rows["wl"].tolist()

        # Build rolling defensive stats by looking up each opponent's pts on same date
        opp_pts_list: list[float] = []
        for _, row in rows.iterrows():
            opp  = row.get("opponent", "")
            date = row["_date"]
            opp_game = self._scores[
                (self._scores["team"] == opp) & (self._scores["_date"] == date)
            ]
            if not opp_game.empty:
                opp_pts_list.append(float(opp_game["team_pts"].iloc[0]))

        rolling_opp_ppg = round(sum(opp_pts_list) / len(opp_pts_list), 1) if opp_pts_list else None

        # Rolling scoring margin (positive = team winning on avg, negative = losing)
        if rolling_ppg is not None and rolling_opp_ppg is not None:
            rolling_margin = round(rolling_ppg - rolling_opp_ppg, 1)
        else:
            rolling_margin = 0.0

        return {
            "wins":             wins,
            "losses":           losses,
            "rolling_ppg":      rolling_ppg,
            "rolling_opp_ppg":  rolling_opp_ppg,
            "rolling_margin":   rolling_margin,
            "wl_list":          wl_list,
            "games":            len(rows),
        }

    def _get_season(self, team: str) -> dict:
        """Current season stats from team_def_df."""
        baseline = 112.0 if self.league == "nba" else 80.0
        pace = 98.0 if self.league == "nba" else 80.0
        default = {"ppg": baseline, "opp_ppg": baseline, "w_pct": 0.5,
                   "pace": pace, "opp_pts_rank": 15 if self.league == "nba" else 7}
        if self._scores.empty:
            return default
        rows = self._scores[self._scores["team"] == team]
        if rows.empty:
            return default
        latest_year = rows["_date"].max().year
        # NBA season begins in autumn; WNBA follows calendar year.
        season_start = pd.Timestamp(year=latest_year if rows["_date"].max().month >= 9 or self.league == "wnba" else latest_year-1,
                                    month=9 if self.league == "nba" else 1, day=1)
        rows = rows[rows["_date"] >= season_start]
        opponents = []
        for _, row in rows.iterrows():
            match = self._scores[(self._scores["team"] == row["opponent"]) & (self._scores["_date"] == row["_date"])]
            if not match.empty:
                opponents.append(float(match.iloc[0]["team_pts"]))
        default.update(ppg=round(float(rows["team_pts"].mean()), 1),
                       opp_ppg=round(sum(opponents)/len(opponents), 1) if opponents else baseline,
                       w_pct=float((rows["wl"] == "W").mean()))
        return default

    def _h2h_boost(self, h2h: dict, home: str, away: str) -> float:
        """
        Small scoring boost based on H2H dominance.
        If one team has won 70%+ of H2H games → +1.0 pt boost for them.
        """
        total = h2h.get("total", 0)
        if total < 2:
            return 0.0
        wins_home = h2h.get("wins_a", 0)   # wins_a = home team wins
        win_rate  = wins_home / total
        if win_rate >= 0.70:
            return 1.0   # home dominates H2H
        if win_rate <= 0.30:
            return -1.0  # away dominates H2H
        return 0.0

    def _build_intel(self, home: str, away: str,
                     hf: dict, af: dict,
                     hs: dict, as_: dict,
                     h2h: dict,
                     ph: float, pa: float, pt: float,
                     home_missing: list | None = None,
                     away_missing: list | None = None) -> list[str]:
        """2-5 data-driven insight bullets (shown in MATCHUP INTEL card)."""
        bullets = []
        home_missing = home_missing or []
        away_missing = away_missing or []

        # Injury alerts — highest priority
        for m in home_missing[:2]:
            bullets.append(f"🚑 {home} missing {m['name']} ({m['ppg']:.0f} PPG, {m['status']})")
        for m in away_missing[:2]:
            bullets.append(f"🚑 {away} missing {m['name']} ({m['ppg']:.0f} PPG, {m['status']})")

        hw, hl = hf.get("wins", 0), hf.get("losses", 0)
        aw, al = af.get("wins", 0), af.get("losses", 0)

        # Form streak
        if hw >= 7:
            bullets.append(f"{home} on a hot streak — {hw}-{hl} in last {hw+hl}")
        if aw >= 7:
            bullets.append(f"{away} on a hot streak — {aw}-{al} in last {aw+al}")

        # Both form records
        if hw + hl >= 5 and aw + al >= 5:
            bullets.append(f"Form L10 — {home}: {hw}-{hl}  |  {away}: {aw}-{al}")

        # H2H record
        total_h2h = h2h.get("total", 0)
        wins_a    = h2h.get("wins_a", 0)
        if total_h2h >= 2:
            bullets.append(f"H2H ({total_h2h} games): {home} leads {wins_a}-{h2h.get('losses_a', 0)}")

        # Defense rank
        hr = hs.get("opp_pts_rank", 15)
        ar = as_.get("opp_pts_rank", 15)
        if hr <= 5:
            bullets.append(f"{home} elite defense — #{hr} in pts allowed ({hs['opp_ppg']:.0f} PPG)")
        elif ar <= 5:
            bullets.append(f"{away} elite defense — #{ar} in pts allowed ({as_['opp_ppg']:.0f} PPG)")

        # Total projection
        if pt < (215 if self.league == "nba" else 155):
            bullets.append(f"Low total projected ({pt:.0f}) — defensive grind expected")
        elif pt > (232 if self.league == "nba" else 175):
            bullets.append(f"High total projected ({pt:.0f}) — uptempo matchup")

        return bullets[:5]
