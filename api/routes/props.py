from fastapi import APIRouter, Query
from typing import Optional

router = APIRouter(tags=["props"])


def _get_props_data():
    from utils.props_cache import get_cached_props
    return get_cached_props()


@router.get("/props")
def get_props(
    game: Optional[str]      = Query(None,    description="Filter by game matchup, e.g. 'MIN @ MEM'"),
    stat: Optional[str]      = Query(None,    description="Stat type: PTS, AST, REB, FG3M, STL, BLK, COMBO"),
    direction: Optional[str] = Query("over",  description="over | under | all"),
    sort: Optional[str]      = Query("ev",    description="ev | hit_rate"),
    limit: int               = Query(100,     ge=1, le=500),
    locks_only: bool         = Query(False),
    combos_only: bool        = Query(False),
):
    cache = _get_props_data()
    props = list(cache.get("main_page_data", []))

    if direction and direction != "all":
        props = [p for p in props if p.get("direction", "").lower() == direction.lower()]

    if game:
        props = [p for p in props if game.lower() in p.get("game_matchup", "").lower()]

    if stat:
        if stat.upper() == "COMBO":
            props = [p for p in props if p.get("is_combo") or "+" in p.get("stat", "")]
        else:
            props = [p for p in props if p.get("stat", "").upper() == stat.upper()]

    if locks_only:
        props = [p for p in props if p.get("is_lock")]

    if combos_only:
        props = [p for p in props if p.get("is_combo") or "+" in p.get("stat", "")]

    key = "ev" if sort == "ev" else "hit_rate"
    props.sort(key=lambda p: (-int(p.get("is_lock", False)), -(p.get(key) or 0)))

    # Slim down the payload — only what the frontend needs
    return {
        "count": len(props),
        "target_date": cache.get("target_date"),
        "game_matchups": cache.get("game_matchups", []),
        "props": [_serialise_prop(p) for p in props[:limit]],
    }


@router.get("/props/alt-lines")
def get_alt_lines():
    cache = _get_props_data()
    return {"alt_lines": cache.get("alt_lines_data", [])}


@router.get("/props/parlays")
def get_parlays():
    from utils.props_cache import get_parlays_cache
    return get_parlays_cache()


def _serialise_prop(p: dict) -> dict:
    insight = p.get("insight", {})
    narrative = (
        insight.get("narrative", "")
        if isinstance(insight, dict)
        else str(insight)
    )
    return {
        "player":      p.get("player", ""),
        "team":        p.get("team", ""),
        "opponent":    p.get("opponent", ""),
        "stat":        p.get("stat", ""),
        "stat_label":  p.get("stat_label", p.get("stat", "")),
        "line":        p.get("line"),
        "avg":         p.get("avg"),
        "l5_avg":      p.get("l5_avg"),
        "direction":   p.get("direction", "Over"),
        "hit_rate":    round(p.get("hit_rate", 0) * 100, 1),
        "hits":        p.get("hits"),
        "total":       p.get("total"),
        "ev":          round(p.get("ev") or 0, 2),
        "is_lock":     bool(p.get("is_lock")),
        "is_combo":    bool(p.get("is_combo")),
        "game_matchup": p.get("game_matchup", ""),
        "blowout_risk": bool(p.get("blowout_risk")),
        "insight":     narrative,
        "def_rank":    p.get("def_rank"),
        "has_live_odds": bool(p.get("has_live_odds")),
        "live_line":   p.get("live_line"),
    }
