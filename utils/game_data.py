"""Explicit, cached game-only ingestion. Never changes player/prop datasets."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STAT_MAP = {'teamScore':'PTS','fieldGoalsMade':'FGM','fieldGoalsAttempted':'FGA',
            'threePointersMade':'FG3M','threePointersAttempted':'FG3A','freeThrowsMade':'FTM',
            'freeThrowsAttempted':'FTA','reboundsOffensive':'OREB','reboundsDefensive':'DREB',
            'reboundsTotal':'REB','assists':'AST','steals':'STL','blocks':'BLK',
            'turnovers':'TOV','numMinutes':'MIN'}


def cached_kaggle_path() -> Path | None:
    """Find an existing mounted cache; never implicitly download an archive."""
    base = Path(os.environ.get('KAGGLEHUB_CACHE', Path.home()/'.cache/kagglehub'))
    base = base/'datasets/eoinamoore/historical-nba-data-and-player-box-scores/versions'
    if not base.exists():
        return None
    paths = sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p:int(p.name), reverse=True)
    return next((p for p in paths if (p/'TeamStatistics.csv').exists() and (p/'Games.csv').exists()), None)


def validate_team_games(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject incomplete targets instead of fabricating opponent scores."""
    if frame.empty:
        return frame.copy()
    d = frame.copy()
    d['GAME_ID'] = d['GAME_ID'].astype(str).str.zfill(10)
    d['GAME_DATE'] = pd.to_datetime(d['GAME_DATE'], errors='raise').dt.strftime('%Y-%m-%d')
    d = d.drop_duplicates(['GAME_ID','TEAM_ABBREVIATION'], keep='last')
    d['PTS'] = pd.to_numeric(d['PTS'], errors='coerce')
    valid_ids=[]
    for game_id, rows in d.groupby('GAME_ID', sort=False):
        if len(rows)!=2 or rows.TEAM_ABBREVIATION.nunique()!=2 or rows.GAME_DATE.nunique()!=1:
            continue
        if rows.PTS.isna().any() or (rows.PTS<=0).any():
            continue
        if 'is_home' in rows and set(rows.is_home.astype(int))!={0,1}:
            continue
        valid_ids.append(game_id)
    d=d[d.GAME_ID.isin(valid_ids)].copy()
    if not d.empty:
        d['OPP_PTS'] = d.groupby('GAME_ID').PTS.transform('sum') - d.PTS
        d['PLUS_MINUS'] = d.PTS-d.OPP_PTS
        d['WL'] = d.PLUS_MINUS.map(lambda n:'W' if n>0 else 'L')
    return d.sort_values(['GAME_DATE','GAME_ID','TEAM_ABBREVIATION']).reset_index(drop=True)


def load_cached_nba(start_date: str = '2020-10-01', cache_path: Path | None = None) -> pd.DataFrame:
    """Read direct team box scores and arenas from a community NBA archive."""
    path = cache_path or cached_kaggle_path()
    if path is None:
        raise FileNotFoundError('No cached TeamStatistics.csv/Games.csv; explicit official refresh required')
    from nba_api.stats.static import teams
    abbreviations = {t['id']:t['abbreviation'] for t in teams.get_teams()}
    raw=pd.read_csv(path/'TeamStatistics.csv', low_memory=False)
    raw=raw[pd.to_datetime(raw.gameDateTimeEst)>=pd.Timestamp(start_date)].copy()
    # Cup final is not a regular-season game. Play-in has distinct semantics.
    raw=raw[raw.gameType.isin(['Regular Season','Playoffs']) & raw.teamId.isin(abbreviations)].copy()
    d=raw.rename(columns=STAT_MAP)
    d['GAME_ID']=raw.gameId.astype(str).str.zfill(10)
    d['GAME_DATE']=pd.to_datetime(raw.gameDateTimeEst).dt.strftime('%Y-%m-%d')
    d['TEAM_ID']=raw.teamId
    d['TEAM_ABBREVIATION']=raw.teamId.map(abbreviations)
    d['OPP_TEAM_ABBREVIATION']=raw.opponentTeamId.map(abbreviations)
    d['is_home']=raw.home.astype(int)
    d['MATCHUP']=d.TEAM_ABBREVIATION + d.is_home.map({1:' vs. ',0:' @ '}) + d.OPP_TEAM_ABBREVIATION
    dates=pd.to_datetime(d.GAME_DATE)
    years=dates.dt.year-(dates.dt.month<7).astype(int)
    d['SEASON']=years.astype(str)+'-'+((years+1)%100).astype(str).str.zfill(2)
    d['SEASON_TYPE']=raw.gameType
    d['SEASON_ID']=d.SEASON_TYPE.map({'Regular Season':'2','Playoffs':'4'})+years.astype(str)
    games=pd.read_csv(path/'Games.csv', usecols=['gameId','arenaName','arenaCity','arenaState'],low_memory=False)
    games['GAME_ID']=games.gameId.astype(str).str.zfill(10)
    d=d.merge(games.drop(columns='gameId'),on='GAME_ID',how='left',validate='many_to_one')
    columns=['GAME_ID','GAME_DATE','TEAM_ID','TEAM_ABBREVIATION','OPP_TEAM_ABBREVIATION','MATCHUP','is_home','SEASON','SEASON_TYPE',*STAT_MAP.values(),'arenaName','arenaCity','arenaState']
    d=d[columns];d['source']=f'kaggle:historical-nba-data-and-player-box-scores:v{path.name}'
    d['fetched_at']=datetime.now(timezone.utc).isoformat()
    return validate_team_games(d)


def fetch_official_team_games(league: str, season: str, season_type: str='Regular Season',
                              timeout: int=20, start_date: str='', end_date: str='') -> pd.DataFrame:
    """One bounded official free stats request; no retry storms or paid calls."""
    if league not in {'nba','wnba'} or season_type not in {'Regular Season','Playoffs'}:
        raise ValueError('Unsupported league or season type')
    from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog
    raw=LeagueGameLog(league_id='00' if league=='nba' else '10',season=season,
                     player_or_team_abbreviation='T',season_type_all_star=season_type,
                     date_from_nullable=start_date,date_to_nullable=end_date,timeout=timeout).get_data_frames()[0]
    raw['SEASON']=season;raw['SEASON_TYPE']=season_type
    raw['is_home']=raw.MATCHUP.str.contains('vs.',regex=False).astype(int)
    raw['OPP_TEAM_ABBREVIATION']=raw.MATCHUP.str.split().str[-1]
    raw['source']='stats.nba.com:LeagueGameLog';raw['fetched_at']=datetime.now(timezone.utc).isoformat()
    return validate_team_games(raw)


def write_history(frame: pd.DataFrame, output: Path) -> dict:
    """Atomically replace history after validation, preserving file on failure."""
    frame=validate_team_games(frame)
    if frame.empty:
        raise ValueError('Refusing to replace history with empty/unpaired games')
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=output.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w') as handle:
            frame.to_csv(handle,index=False)
        os.replace(temporary,output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    report={'rows':len(frame),'games':frame.GAME_ID.nunique(),'first_game':frame.GAME_DATE.min(),
            'last_game':frame.GAME_DATE.max(),'written_at':datetime.now(timezone.utc).isoformat(),
            'sources':sorted(frame.source.dropna().unique()) if 'source' in frame else [],
            'historical_pregame_injuries':False,
            'note':'Freshness must be compared to latest completed league game; fetch time does not imply current coverage.'}
    report_path=output.with_suffix('.metadata.json')
    fd,temporary=tempfile.mkstemp(dir=output.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w') as handle:json.dump(report,handle,indent=2)
        os.replace(temporary,report_path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return report


def load_game_team_history(league: str) -> pd.DataFrame:
    """Read isolated validated team history, normalized for pregame features.

    Does not fetch implicitly. A missing file is an actionable data error,
    rather than silently reverting to incomplete player-derived targets.
    """
    if league not in {'nba','wnba'}:
        raise ValueError('Unsupported league')
    path=Path(__file__).resolve().parents[1]/'data'/league/'game_team_history.csv'
    raw=pd.read_csv(path,dtype={'GAME_ID':str})
    from utils.game_features import normalize_team_games
    return normalize_team_games(raw,league)
