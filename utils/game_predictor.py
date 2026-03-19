# utils/game_predictor.py
"""
Game-Level Predictor — Spread, Total, and Winner
==================================================
Uses team recent form (rolling 10 games) + season defensive ratings
+ home court advantage + pace + head-to-head history.

Form is the primary driver of predictions — recent momentum matters
more than season-long averages.
"""

import logging
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


class GamePredictor:
    """
    Predicts winner, spread, and total using:
      - Rolling 10-game team PPG (from player game logs)
      - Team defensive ratings (OPP_PTS from team_def_df)
      - Recent W/L form momentum
      - Head-to-head record
      - Home court advantage + pace adjustment
    """

    def __init__(self, team_def_df: pd.DataFrame, player_logs_df: pd.DataFrame):
        self.team_def = team_def_df
        self.logs     = player_logs_df
        self._scores  = self._build_scores(player_logs_df)

    def refresh(self, player_logs_df: pd.DataFrame) -> None:
        """Re-build the game scores table from a fresh DF (call after data update)."""
        self.logs    = player_logs_df
        self._scores = self._build_scores(player_logs_df)

    # ── Public API ────────────────────────────────────────────────────────

    def predict_game(self, home: str, away: str,
                     home_injuries: list | None = None,
                     away_injuries: list | None = None) -> dict:
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
        home_form   = self._get_form(home)
        away_form   = self._get_form(away)
        home_season = self._get_season(home)
        away_season = self._get_season(away)
        h2h         = self.get_h2h(home, away)

        # ── Offensive estimate per team ───────────────────────────────────
        home_off_raw = home_form.get("rolling_ppg") or home_season.get("ppg", 112.0)
        away_off_raw = away_form.get("rolling_ppg") or away_season.get("ppg", 112.0)

        home_def_raw = home_season.get("opp_ppg", 112.0)
        away_def_raw = away_season.get("opp_ppg", 112.0)

        # Blend form vs season defense
        pred_home = (home_off_raw * FORM_WEIGHT + away_def_raw * SEASON_WEIGHT)
        pred_away = (away_off_raw * FORM_WEIGHT + home_def_raw * SEASON_WEIGHT)

        # Form momentum
        home_form_wp   = home_form.get("wins", 0) / max(home_form.get("wins", 0) + home_form.get("losses", 0), 1)
        away_form_wp   = away_form.get("wins", 0) / max(away_form.get("wins", 0) + away_form.get("losses", 0), 1)
        home_season_wp = home_season.get("w_pct", 0.5)
        away_season_wp = away_season.get("w_pct", 0.5)
        pred_home += (home_form_wp - home_season_wp) * 4.0
        pred_away += (away_form_wp - away_season_wp) * 4.0

        # H2H boost
        h2h_boost = self._h2h_boost(h2h, home, away)
        pred_home += h2h_boost
        pred_away -= h2h_boost

        # Home court advantage
        pred_home += HOME_COURT_ADV

        # Pace adjustment
        home_pace   = home_season.get("pace", LEAGUE_AVG_PACE)
        away_pace   = away_season.get("pace", LEAGUE_AVG_PACE)
        pace_factor = ((home_pace + away_pace) / 2) / LEAGUE_AVG_PACE
        pred_home   = round(pred_home * pace_factor, 1)
        pred_away   = round(pred_away * pace_factor, 1)

        # ── Injury adjustments ───────────────────────────────────────────
        home_missing, away_missing = [], []
        if home_injuries:
            pred_home, home_missing = self._apply_injury_penalty(pred_home, home_injuries)
        if away_injuries:
            pred_away, away_missing = self._apply_injury_penalty(pred_away, away_injuries)

        pred_total  = round(pred_home + pred_away, 1)
        pred_spread = round(pred_home - pred_away, 1)

        margin = abs(pred_spread)
        winner_conf = "HIGH" if margin >= 8 else "MEDIUM" if margin >= 4 else "LOW"
        winner      = home if pred_home >= pred_away else away

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

        return {
            "predicted_home_score": pred_home,
            "predicted_away_score": pred_away,
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
            edge = model_spread - actual_spread
            if edge > MIN_EDGE_SPREAD:
                result["spread_pick"] = "HOME"
                result["spread_team"] = home
            elif edge < -MIN_EDGE_SPREAD:
                result["spread_pick"] = "AWAY"
                result["spread_team"] = away
            abs_edge = abs(edge)
            result["spread_confidence"] = (
                "HIGH" if abs_edge >= 4.0 else "MEDIUM" if abs_edge >= 2.0 else "LOW"
            )

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

    def _apply_injury_penalty(self, raw_score: float, injuries: list) -> tuple:
        """
        Deduct estimated scoring contribution of OUT/DOUBTFUL players.

        Returns (adjusted_score, missing_list)
        missing_list items: {"name": str, "ppg": float, "status": str}
        """
        penalty  = 0.0
        missing  = []
        for player in injuries:
            status = player.get("status", "")
            if status not in ("OUT", "DOUBTFUL"):
                continue
            name     = player.get("name", "")
            # Look up player's L10 PPG
            player_rows = self.logs[self.logs["PLAYER_NAME"] == name]
            if len(player_rows) < 3:
                continue
            ppg = float(
                player_rows.sort_values("_date", ascending=False)
                .head(10)["PTS"]
                .mean()
            )
            if ppg < 5:           # skip non-scorers
                continue
            weight   = 0.70 if status == "OUT" else 0.35
            penalty += ppg * weight
            missing.append({"name": name, "ppg": round(ppg, 1), "status": status})

        # Floor: never reduce below 75 % of the original projection
        adjusted = max(raw_score - penalty, raw_score * 0.75)
        return round(adjusted, 1), missing

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

        winner_reason = (
            f"{winner} projected to win {ph:.0f}–{pa:.0f} "
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
        home_pace = hs.get("pace", LEAGUE_AVG_PACE)
        away_pace = as_.get("pace", LEAGUE_AVG_PACE)
        avg_pace  = (home_pace + away_pace) / 2
        if avg_pace > LEAGUE_AVG_PACE + 2:
            pace_note = f" Both teams play fast (avg pace {avg_pace:.0f} vs league {LEAGUE_AVG_PACE:.0f})."
        elif avg_pace < LEAGUE_AVG_PACE - 2:
            pace_note = f" Slow-pace matchup (avg {avg_pace:.0f})."

        total_reason = (
            f"Model projects {pt:.0f} combined pts "
            f"({home} {ph:.0f} + {away} {pa:.0f})."
            f" {home} allows {hs.get('opp_ppg', 0):.0f} PPG, "
            f"{away} allows {as_.get('opp_ppg', 0):.0f} PPG.{pace_note}"
        )

        # ── Spread reason ─────────────────────────────────────────────────
        home_def_rank = hs.get("opp_pts_rank", 15)
        away_def_rank = as_.get("opp_pts_rank", 15)
        spread_reason = (
            f"Model spread: {home} {ps:+.1f}. "
            f"{home} defense #{home_def_rank} (allows {hs.get('opp_ppg', 0):.0f} PPG), "
            f"{away} defense #{away_def_rank} (allows {as_.get('opp_ppg', 0):.0f} PPG). "
            f"H2H: {home} leads {h2h.get('wins_a', 0)}-{h2h.get('losses_a', 0)} "
            f"in {h2h.get('total', 0)} meetings."
        ) if h2h.get("total", 0) >= 1 else (
            f"Model spread: {home} {ps:+.1f}. "
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
        """Rolling form for last n games."""
        if self._scores.empty:
            return {"wins": 0, "losses": 0, "rolling_ppg": None, "wl_list": [], "games": 0}

        rows = (
            self._scores[self._scores["team"] == team]
            .sort_values("_date", ascending=False)
            .head(n)
        )

        if rows.empty:
            return {"wins": 0, "losses": 0, "rolling_ppg": None, "wl_list": [], "games": 0}

        wins   = int((rows["wl"] == "W").sum())
        losses = int((rows["wl"] == "L").sum())
        rolling_ppg = round(float(rows["team_pts"].mean()), 1) if len(rows) >= 3 else None
        wl_list = rows["wl"].tolist()

        return {
            "wins":        wins,
            "losses":      losses,
            "rolling_ppg": rolling_ppg,
            "wl_list":     wl_list,
            "games":       len(rows),
        }

    def _get_season(self, team: str) -> dict:
        """Current season stats from team_def_df."""
        default = {"ppg": 112.0, "opp_ppg": 112.0, "w_pct": 0.5,
                   "pace": LEAGUE_AVG_PACE, "opp_pts_rank": 15}

        if self.team_def.empty or "TEAM_ABBREVIATION" not in self.team_def.columns:
            return default

        rows = self.team_def[self.team_def["TEAM_ABBREVIATION"] == team]
        if rows.empty:
            return default

        row = rows.sort_values("SEASON", ascending=False).iloc[0]
        opp_pts    = float(row.get("OPP_PTS",    112.0))
        plus_minus = float(row.get("PLUS_MINUS",   0.0))
        w_pct      = float(row.get("W_PCT",         0.5))
        pace       = float(row["PACE"]) if "PACE" in row.index else LEAGUE_AVG_PACE
        opp_rank   = int(row.get("OPP_PTS_RANK", 15))

        return {
            "ppg":          round(opp_pts + plus_minus, 1),
            "opp_ppg":      round(opp_pts, 1),
            "w_pct":        round(w_pct, 3),
            "pace":         round(pace, 1),
            "opp_pts_rank": opp_rank,
        }

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
        if pt < 215:
            bullets.append(f"Low total projected ({pt:.0f}) — defensive grind expected")
        elif pt > 232:
            bullets.append(f"High total projected ({pt:.0f}) — uptempo matchup")

        return bullets[:5]
