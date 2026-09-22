from fastapi import APIRouter, Path, HTTPException
from typing import Optional
from fastapi import Query

router = APIRouter(tags=["players"])

_DF_CACHE: dict = {}


def _get_df():
    if not _DF_CACHE:
        from utils.kaggle_loader import load_player_game_logs
        _DF_CACHE["df"] = load_player_game_logs()
    return _DF_CACHE["df"]


@router.get("/players")
def list_players(q: Optional[str] = Query(None, description="Search substring")):
    df = _get_df()
    if df.empty:
        return {"players": []}
    names = sorted(df["PLAYER_NAME"].unique().tolist())
    if q:
        names = [n for n in names if q.lower() in n.lower()]
    return {"players": names[:200]}


@router.get("/player/{player_name}/chart-data")
def get_player_chart_data(
    player_name: str = Path(..., description="Player name"),
    stat: str       = Query("PTS", description="Stat column, e.g. PTS, AST, REB"),
    games: int      = Query(20, ge=5, le=82),
):
    df = _get_df()
    if df.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")

    player_df = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    if player_df.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    stat_col = stat.upper()
    if stat_col not in player_df.columns:
        raise HTTPException(status_code=400, detail=f"Unknown stat '{stat}'")

    player_df = player_df.sort_values("GAME_DATE").tail(games)
    line = None
    try:
        from utils.props_cache import get_cached_props
        cache = get_cached_props()
        for p in cache.get("main_page_data", []):
            if p.get("player", "").lower() == player_name.lower() and p.get("stat", "").upper() == stat_col:
                line = p.get("line")
                break
    except Exception:
        pass

    avg = round(float(player_df[stat_col].mean()), 1) if not player_df.empty else None
    l5_avg = round(float(player_df[stat_col].tail(5).mean()), 1) if len(player_df) >= 5 else avg

    records = []
    for _, row in player_df.iterrows():
        date = str(row.get("GAME_DATE", ""))[:10]
        opp  = row.get("MATCHUP", "")
        val  = row.get(stat_col)
        records.append({
            "date":     date,
            "opponent": opp,
            "value":    round(float(val), 1) if val is not None else None,
            "hit":      bool(val >= line) if (line is not None and val is not None) else None,
        })

    return {
        "player": player_name,
        "stat":   stat_col,
        "line":   line,
        "avg":    avg,
        "l5_avg": l5_avg,
        "games":  records,
    }


@router.get("/player/{player_name}/stats")
def get_player_stats(player_name: str = Path(...)):
    df = _get_df()
    if df.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")

    player_df = df[df["PLAYER_NAME"].str.lower() == player_name.lower()]
    if player_df.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    stat_cols = [c for c in ["PTS", "AST", "REB", "FG3M", "STL", "BLK"] if c in player_df.columns]
    season_avgs = {col: round(float(player_df[col].mean()), 1) for col in stat_cols}
    l5 = player_df.sort_values("GAME_DATE").tail(5)
    l5_avgs = {col: round(float(l5[col].mean()), 1) for col in stat_cols}

    team = player_df.sort_values("GAME_DATE").iloc[-1].get("TEAM_ABBREVIATION", "")

    try:
        from utils.kaggle_loader import load_player_positions
        pos_df = load_player_positions()
        pos_row = pos_df[pos_df["PLAYER_NAME"].str.lower() == player_name.lower()]
        position = pos_row.iloc[-1]["POSITION"] if not pos_row.empty else ""
    except Exception:
        position = ""

    return {
        "player":       player_name,
        "team":         str(team),
        "position":     position,
        "season_avgs":  season_avgs,
        "l5_avgs":      l5_avgs,
        "games_played": int(len(player_df)),
    }
