"""WNBA data loader — pulls player and team game logs from nba_api and
exports pipeline-ready CSVs to data/wnba/.

The WNBA path is intentionally separate from utils/kaggle_loader.py because:
- WNBA data source is nba_api (not Kaggle) — see docs/wnba-data-source.md
- The pull is fast enough (<1s per season) that caching an archive isn't needed
- Schema from LeagueGameLog already matches EXPECTED_PLAYER_LOG_COLUMNS closely
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils.league_config import get_config

logger = logging.getLogger(__name__)

_WNBA = get_config("wnba")
_LEAGUE_ID = _WNBA.api_league_id  # "10"


# =============================================================================
# SEASON HELPERS
# =============================================================================


def get_recent_wnba_seasons(num_seasons: int = 3) -> list[str]:
    """Get last N WNBA seasons as string years, e.g. ['2025', '2024', '2023'].

    WNBA season = calendar year (May–Oct). If we're past May, current year
    counts; otherwise the previous year.
    """
    today = datetime.now()
    start_year = today.year if today.month >= _WNBA.season_start_month else today.year - 1
    return [str(start_year - i) for i in range(num_seasons)]


# =============================================================================
# RAW GAME LOGS FROM nba_api
# =============================================================================


def _fetch_season_player_logs(season: str) -> pd.DataFrame:
    """Fetch WNBA player game logs for one season via nba_api."""
    from nba_api.stats.endpoints import leaguegamelog

    log = leaguegamelog.LeagueGameLog(
        season=season,
        league_id=_LEAGUE_ID,
        player_or_team_abbreviation="P",
    )
    df = log.get_data_frames()[0]
    df["SEASON"] = season
    return df


def _fetch_season_team_logs(season: str) -> pd.DataFrame:
    """Fetch WNBA team game logs for one season."""
    from nba_api.stats.endpoints import leaguegamelog

    log = leaguegamelog.LeagueGameLog(
        season=season,
        league_id=_LEAGUE_ID,
        player_or_team_abbreviation="T",
    )
    df = log.get_data_frames()[0]
    df["SEASON"] = season
    return df


def fetch_player_game_logs(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """Fetch WNBA player game logs across multiple seasons and concatenate.

    Returns DataFrame in the shape that WNBA callers (feature engineering,
    dashboard) expect: normalized column names matching the NBA pipeline.
    """
    if seasons is None:
        seasons = get_recent_wnba_seasons(3)

    frames = []
    for season in seasons:
        t0 = time.time()
        df = _fetch_season_player_logs(season)
        logger.info(f"WNBA {season}: {len(df)} player-game rows ({time.time()-t0:.1f}s)")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    return _normalize_player_logs(raw)


def _normalize_player_logs(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename/reshape raw nba_api columns to match the pipeline schema."""
    # nba_api LeagueGameLog columns already include: SEASON_ID, PLAYER_ID,
    # PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID, GAME_DATE,
    # MATCHUP, WL, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA,
    # FT_PCT, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, PLUS_MINUS,
    # FANTASY_PTS, VIDEO_AVAILABLE

    df = raw.rename(columns={"PLAYER_ID": "Player_ID", "GAME_ID": "Game_ID"})
    # NBA pipeline uses "Mmm dd, YYYY" — reformat WNBA "YYYY-MM-DD" to match
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%b %d, %Y")
    if "FANTASY_PTS" in df.columns:
        df = df.drop(columns=["FANTASY_PTS"])
    return df


# =============================================================================
# TEAM STATS AGGREGATION
# =============================================================================


def load_team_stats(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """Aggregate WNBA team-level stats across seasons.

    Returns per-team-per-season averages plus PACE, matching the shape of the
    NBA `team_stats.csv` schema loosely (used by matchup card rankings).
    """
    if seasons is None:
        seasons = get_recent_wnba_seasons(3)

    frames = []
    for season in seasons:
        df = _fetch_season_team_logs(season)
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # Regular season only: WNBA nba_api SEASON_ID prefix "2" = regular season
    if "SEASON_ID" in raw.columns:
        raw = raw[raw["SEASON_ID"].astype(str).str.startswith("2")]

    # Win/loss flag for aggregation
    raw["W"] = (raw["WL"] == "W").astype(int)

    agg = raw.groupby(["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "SEASON"], as_index=False).agg(
        GP=("GAME_ID", "count"),
        W=("W", "sum"),
        MIN=("MIN", "mean"),
        FGM=("FGM", "mean"),
        FGA=("FGA", "mean"),
        FG3M=("FG3M", "mean"),
        FG3A=("FG3A", "mean"),
        FTM=("FTM", "mean"),
        FTA=("FTA", "mean"),
        OREB=("OREB", "mean"),
        DREB=("DREB", "mean"),
        REB=("REB", "mean"),
        AST=("AST", "mean"),
        TOV=("TOV", "mean"),
        STL=("STL", "mean"),
        BLK=("BLK", "mean"),
        PF=("PF", "mean"),
        PTS=("PTS", "mean"),
        PLUS_MINUS=("PLUS_MINUS", "mean"),
    )

    agg["L"] = agg["GP"] - agg["W"]
    agg["W_PCT"] = (agg["W"] / agg["GP"]).round(3)
    agg["FG_PCT"] = (agg["FGM"] / agg["FGA"]).round(3)
    agg["FG3_PCT"] = (agg["FG3M"] / agg["FG3A"]).round(3)
    agg["FT_PCT"] = (agg["FTM"] / agg["FTA"]).round(3)
    agg["PACE"] = (48 * (agg["FGA"] + 0.44 * agg["FTA"] - agg["OREB"] + agg["TOV"]) / (agg["MIN"] / 5)).round(1)

    avg_cols = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF",
                "PTS", "PLUS_MINUS"]
    for c in avg_cols:
        agg[c] = agg[c].round(1)

    agg = agg.rename(columns={"TEAM_ABBREVIATION": "TEAM_ABBREVIATION", "TEAM_NAME": "TEAM_NAME"})
    return agg


# =============================================================================
# PLAYER POSITIONS (inferred from stats)
# =============================================================================


def load_player_positions(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Infer player positions from stat lines. WNBA-specific thresholds.

    Returns columns: PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, POSITION, SEASON.
    """
    thresholds = _WNBA.position_thresholds
    guard_min_ast = thresholds["guard_min_ast"]
    center_min_reb = thresholds["center_min_reb"]

    stats = player_logs.copy()
    if "Player_ID" not in stats.columns:
        stats = stats.rename(columns={"PLAYER_ID": "Player_ID"})

    avgs = stats.groupby(["Player_ID", "SEASON"], as_index=False).agg(
        AST_AVG=("AST", "mean"),
        REB_AVG=("REB", "mean"),
    )

    def _infer(row):
        if row["AST_AVG"] >= guard_min_ast and row["REB_AVG"] < 5.0:
            return "G"
        if row["REB_AVG"] >= center_min_reb:
            return "C"
        return "F"

    avgs["POSITION"] = avgs.apply(_infer, axis=1)

    # Most recent team per player per season
    stats_sorted = stats.sort_values("GAME_DATE", ascending=False)
    latest = stats_sorted.drop_duplicates(subset=["Player_ID", "SEASON"], keep="first")
    latest = latest[["Player_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "SEASON"]]

    result = latest.merge(avgs[["Player_ID", "SEASON", "POSITION"]], on=["Player_ID", "SEASON"], how="left")
    result["POSITION"] = result["POSITION"].fillna("F")
    result = result.rename(columns={"Player_ID": "PLAYER_ID"})
    return result[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "POSITION", "SEASON"]]


# =============================================================================
# TEAM DEFENSIVE STATS (opponent scoring/shooting when facing this team)
# =============================================================================


def load_team_defensive_stats(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """Aggregate opponent stats faced by each WNBA team across seasons.

    Mirrors the NBA `team_defensive_stats.csv` shape. When teams play a game,
    each row in `_fetch_season_team_logs` represents one team's performance;
    we join back on gameId to get the opposing team's stats and aggregate
    those by TEAM_ID (the team being defended).
    """
    if seasons is None:
        seasons = get_recent_wnba_seasons(3)

    frames = []
    for season in seasons:
        df = _fetch_season_team_logs(season)
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # Self-join on GAME_ID to pair each team's row with its opponent's row
    left = raw.rename(columns={c: c for c in raw.columns})
    right = raw.rename(columns={c: f"OPP_{c}" for c in raw.columns if c != "GAME_ID"})
    joined = left.merge(right, on="GAME_ID")
    # Keep only rows where TEAM_ID != OPP_TEAM_ID (each game -> 2 rows)
    joined = joined[joined["TEAM_ID"] != joined["OPP_TEAM_ID"]]

    # For each defending team, average opponent's offensive line
    agg = joined.groupby(
        ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "SEASON"], as_index=False
    ).agg(
        GP=("GAME_ID", "count"),
        OPP_PTS=("OPP_PTS", "mean"),
        OPP_FGM=("OPP_FGM", "mean"),
        OPP_FGA=("OPP_FGA", "mean"),
        OPP_FG3M=("OPP_FG3M", "mean"),
        OPP_FG3A=("OPP_FG3A", "mean"),
        OPP_FTM=("OPP_FTM", "mean"),
        OPP_FTA=("OPP_FTA", "mean"),
        OPP_OREB=("OPP_OREB", "mean"),
        OPP_DREB=("OPP_DREB", "mean"),
        OPP_REB=("OPP_REB", "mean"),
        OPP_AST=("OPP_AST", "mean"),
        OPP_TOV=("OPP_TOV", "mean"),
        OPP_STL=("OPP_STL", "mean"),
        OPP_BLK=("OPP_BLK", "mean"),
        OPP_PF=("OPP_PF", "mean"),
    )

    agg["OPP_FG_PCT"] = (agg["OPP_FGM"] / agg["OPP_FGA"]).round(3)
    agg["OPP_FG3_PCT"] = (agg["OPP_FG3M"] / agg["OPP_FG3A"]).round(3)
    agg["OPP_FT_PCT"] = (agg["OPP_FTM"] / agg["OPP_FTA"]).round(3)

    for c in ["OPP_PTS", "OPP_FGM", "OPP_FGA", "OPP_FG3M", "OPP_FG3A",
              "OPP_FTM", "OPP_FTA", "OPP_OREB", "OPP_DREB", "OPP_REB",
              "OPP_AST", "OPP_TOV", "OPP_STL", "OPP_BLK", "OPP_PF"]:
        agg[c] = agg[c].round(1)

    # Rank columns — lower is better on defense (opponents score less)
    for col in ["OPP_PTS", "OPP_FG_PCT", "OPP_FG3_PCT", "OPP_REB", "OPP_AST"]:
        agg[f"{col}_RANK"] = agg.groupby("SEASON")[col].rank(ascending=True, method="min").astype(int)

    return agg


# =============================================================================
# DEFENSE VS POSITION (opponent scoring by defender position)
# =============================================================================


def load_defense_vs_position(
    player_logs: pd.DataFrame,
    positions: pd.DataFrame,
    seasons: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate what each team allows opponents to score, split by position.

    Returns columns: TEAM_ABBREVIATION, SEASON, POSITION, PTS, AST, REB, FG3M,
    plus per-season rank columns (lower = stronger defense vs that position).
    """
    if player_logs.empty or positions.empty:
        return pd.DataFrame()

    if seasons is None:
        seasons = get_recent_wnba_seasons(3)
    seasons_set = set(str(s) for s in seasons)

    logs = player_logs.copy()
    if "Player_ID" not in logs.columns and "PLAYER_ID" in logs.columns:
        logs = logs.rename(columns={"PLAYER_ID": "Player_ID"})
    logs["SEASON"] = logs["SEASON"].astype(str)
    logs = logs[logs["SEASON"].isin(seasons_set)]

    pos = positions.rename(columns={"PLAYER_ID": "Player_ID"})[["Player_ID", "SEASON", "POSITION"]].copy()
    pos["SEASON"] = pos["SEASON"].astype(str)

    merged = logs.merge(pos, on=["Player_ID", "SEASON"], how="left")
    merged["POSITION"] = merged["POSITION"].fillna("F")

    # Opponent tricode from MATCHUP: "LVA vs. ATL" -> ATL, "LVA @ NYL" -> NYL
    def _opp(matchup):
        if not isinstance(matchup, str):
            return ""
        parts = matchup.split()
        return parts[-1] if parts else ""

    merged["OPP_TEAM"] = merged["MATCHUP"].apply(_opp)
    merged = merged[merged["OPP_TEAM"] != ""]

    agg = merged.groupby(["OPP_TEAM", "SEASON", "POSITION"], as_index=False).agg(
        GP=("Game_ID", "count"),
        PTS=("PTS", "mean"),
        AST=("AST", "mean"),
        REB=("REB", "mean"),
        FG3M=("FG3M", "mean"),
    )
    for c in ["PTS", "AST", "REB", "FG3M"]:
        agg[c] = agg[c].round(1)

    # Rank within (SEASON, POSITION) — rank 1 = allows fewest (best defense)
    for stat in ["PTS", "AST", "REB", "FG3M"]:
        agg[f"{stat}_RANK"] = (
            agg.groupby(["SEASON", "POSITION"])[stat]
            .rank(ascending=True, method="min").astype(int)
        )

    return agg.rename(columns={"OPP_TEAM": "TEAM_ABBREVIATION"})


# =============================================================================
# EXPORT ALL WNBA PIPELINE CSVs
# =============================================================================


def export_pipeline_csvs(seasons: Iterable[str] | None = None) -> None:
    """Load all WNBA data and write pipeline-ready CSVs to data/wnba/."""
    data_dir = _WNBA.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if seasons is None:
        seasons = get_recent_wnba_seasons(3)
    seasons = list(seasons)

    print(f"[WNBA] Fetching player game logs for {seasons}...")
    player_logs = fetch_player_game_logs(seasons)
    player_logs.to_csv(data_dir / "player_game_logs.csv", index=False)
    print(f"  -> {len(player_logs)} rows, {player_logs['PLAYER_NAME'].nunique()} players")

    print("[WNBA] Loading team stats...")
    team_stats = load_team_stats(seasons)
    team_stats.to_csv(data_dir / "team_stats.csv", index=False)
    print(f"  -> {len(team_stats)} rows")

    print("[WNBA] Loading team defensive stats...")
    team_def = load_team_defensive_stats(seasons)
    team_def.to_csv(data_dir / "team_defensive_stats.csv", index=False)
    print(f"  -> {len(team_def)} rows")

    print("[WNBA] Inferring player positions...")
    positions = load_player_positions(player_logs)
    positions.to_csv(data_dir / "player_positions.csv", index=False)
    print(f"  -> {len(positions)} rows")

    print("[WNBA] Computing defense vs position...")
    def_vs_pos = load_defense_vs_position(player_logs, positions, seasons)
    def_vs_pos.to_csv(data_dir / "defense_vs_position.csv", index=False)
    print(f"  -> {len(def_vs_pos)} rows")

    # Build the full engineered parquet (models depend on all features being present)
    print("[WNBA] Building engineered_data.parquet with full feature set...")
    try:
        from utils.feature_engineering import engineer_features
        # Positions has per-season rows; dedupe to latest per player so
        # add_position_features doesn't multiply games.
        positions_deduped = (
            positions.sort_values("SEASON", ascending=False)
            .drop_duplicates(subset=["PLAYER_ID"], keep="first")
        )
        enriched = engineer_features(
            player_logs,
            team_defensive_stats=team_def,
            team_stats=team_stats,
            player_positions=positions_deduped,
            defense_vs_position=def_vs_pos,
        )
        enriched["_date"] = pd.to_datetime(enriched["GAME_DATE"], format="%b %d, %Y", errors="coerce")
        enriched.to_parquet(data_dir / "engineered_data.parquet", index=False)
        print(f"  -> {len(enriched)} rows, {len(enriched.columns)} columns")
    except Exception as e:
        print(f"  ! parquet build failed: {e}")

    print("[WNBA] Pipeline CSVs exported to data/wnba/")
