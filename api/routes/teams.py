"""Team profiles and published ESPN game highlights."""
from fastapi import APIRouter, Path, Query
from api.data import League
from utils.basketball_content import get_teams, get_highlights
from utils.basketball_standings import get_standings
from utils.basketball_boxscore import get_boxscore

router = APIRouter(tags=['teams', 'highlights'])


@router.get('/standings')
def standings(league: League = 'nba', season: int | None = Query(None, ge=1997, le=2100)):
    return get_standings(league, season)


@router.get('/games/{game_id}/boxscore')
def boxscore(game_id: str = Path(pattern=r'^\d{6,12}$'), league: League = 'nba'):
    return get_boxscore(league, game_id)


@router.get('/teams')
def teams(league: League = 'nba', season: int | None = Query(None, ge=1997, le=2100)):
    return get_teams(league, season)


@router.get('/teams/{abbrev}')
def team(abbrev: str, league: League = 'nba', season: int | None = Query(None, ge=1997, le=2100)):
    return get_teams(league, season, abbrev)


@router.get('/highlights')
def highlights(league: League = 'nba'):
    return get_highlights(league)
