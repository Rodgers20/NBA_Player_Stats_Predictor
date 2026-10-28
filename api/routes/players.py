from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Literal
import logging
import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from api.data import League, history, with_q1, predictor, number
from functools import lru_cache
from utils.game_context import annotate, build_league_context
from utils.league_config import get_config

from utils.quarter_stats import Q1_STATS

router = APIRouter(tags=['players'])
logger = logging.getLogger(__name__)
STATS = {'PTS', 'AST', 'REB', 'FG3M', 'STL', 'BLK', 'PTS+REB', 'PTS+AST', 'REB+AST', 'PTS+REB+AST', 'STL+BLK',
         'PTS_1Q', 'REB_1Q', 'AST_1Q', 'FG3M_1Q'}


def _headshot_url(frame: pd.DataFrame, league: League) -> str | None:
    """Use the same league CDN and game-log player ID as the Dash dashboard."""
    id_column = next((column for column in ('Player_ID', 'PLAYER_ID') if column in frame), None)
    if id_column is None:
        return None
    ids = pd.to_numeric(frame[id_column], errors='coerce').dropna()
    if ids.empty:
        return None
    return get_config(league).headshot_cdn_template.format(player_id=int(ids.iloc[-1]))


def _injury_context(player_name: str, league: League) -> tuple[str | None, str | None]:
    """Return only a reported status; missing news is not proof of availability."""
    try:
        if league == 'wnba':
            from utils.wnba_injuries import get_wnba_player_injury
            injury = get_wnba_player_injury(player_name)
        else:
            from utils.injury_news import get_player_injury_status
            injury = get_player_injury_status(player_name)
            confidence = number(injury.get('confidence')) if injury else None
            if confidence is None or confidence < 0.7:
                injury = None
    except Exception:
        logger.exception('Player injury status unavailable: %s %s', league, player_name)
        injury = None
    if not injury:
        return None, None
    status = str(injury.get('status') or '').strip().upper() or None
    reason = str(injury.get('reason') or '').strip() or None
    if reason in ('No injury news found', 'No clear injury indicators'):
        reason = None
    return status, reason


@lru_cache(maxsize=4)
def _read_team_history(path, modified):
    return pd.read_csv(path, usecols=['GAME_ID', 'TEAM_ABBREVIATION', 'PTS', 'OPP_PTS'])


def _team_history(league):
    """Team game scores, or None when this league has no team-game history."""
    path = get_config(league).data_dir / 'game_team_history.csv'
    return _read_team_history(path, path.stat().st_mtime_ns) if path.exists() else None


_context_cache = {}


def _league_context(league):
    """Indexed once per loaded history; the cache holds the frame itself so its id cannot be reused."""
    frame = history(league)
    cached = _context_cache.get(league)
    if cached is None or cached[0] is not frame:
        _context_cache[league] = cached = (frame, build_league_context(frame, _team_history(league)))
    return cached[1]


def _game_context(frame, league):
    """Per-game minutes, margin, result, rest and absent teammates for the player's rows; [] when the logs lack them."""
    try:
        return annotate(frame, _league_context(league))
    except (KeyError, ValueError, TypeError):
        logger.warning('Game context unavailable for %s history', league, exc_info=True)
        return []


def _context_fields(context, size):
    """The last `size` context records as sparse dicts: missing values are left out to keep the payload small."""
    rows = context[-size:] if size and len(context) >= size else [{}] * size
    return [{key: value for key, value in row.items() if value is not None and value != []} for row in rows]


def _player(player_name, league):
    df = history(league)
    frame = df[df['PLAYER_NAME'].str.casefold() == player_name.casefold()].copy()
    if frame.empty:
        raise HTTPException(404, f"Player '{player_name}' not found")
    if {'Game_ID', 'Player_ID'} <= set(frame):  # older or test logs without ids simply have no first quarters
        frame = with_q1(frame, league)
    today = pd.Timestamp(datetime.now(ZoneInfo('America/New_York')).date())
    return frame[frame['_date'] < today].sort_values('_date')


@router.get('/players')
def list_players(q: str | None = None, league: League = 'nba'):
    frame = history(league)
    names = sorted(frame['PLAYER_NAME'].dropna().unique().tolist())
    if q:
        names = [name for name in names if q.casefold() in name.casefold()]
    # One photo per name for search lists, from the player's most recent game id (same CDN as the player page).
    id_column = next((column for column in ('Player_ID', 'PLAYER_ID') if column in frame), None)
    headshots = {}
    if id_column:
        ids = frame.assign(_id=pd.to_numeric(frame[id_column], errors='coerce')).dropna(subset=['_id']).groupby('PLAYER_NAME')['_id'].last()
        template = get_config(league).headshot_cdn_template
        headshots = {name: template.format(player_id=int(ids[name])) for name in names if name in ids}
    return {'players': names, 'headshots': headshots}


@router.get('/players/scheduled')
def list_scheduled_players(game_date: date, league: League = 'nba'):
    from utils.scheduled_players import scheduled_players
    return scheduled_players(league, game_date.isoformat())


@router.get('/player/{player_name}/series')
def get_player_series(player_name: str, games: int = Query(200, ge=5, le=200),
                      league: League = 'nba'):
    """One compact, chronological game series for every static chart statistic."""
    whole = _player(player_name, league)
    frame = whole.tail(games)
    base_stats = ('PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK')
    missing = [stat for stat in base_stats if stat not in frame]
    if missing:
        raise HTTPException(503, f'Player history lacks {", ".join(missing)}')
    context = _context_fields(_game_context(whole, league), len(frame))
    records = []
    for (_, row), extra in zip(frame.iterrows(), context):
        matchup = str(row.get('MATCHUP') or '')
        is_home = True if ' vs. ' in matchup else False if ' @ ' in matchup else None
        if is_home is None and 'is_home' in frame:
            home_value = number(row.get('is_home'))
            if home_value in (0, 1):
                is_home = bool(home_value)
        season = row.get('SEASON')
        records.append(dict(date=row['_date'].date().isoformat(), opponent=matchup,
                            season=None if pd.isna(season) else str(season), is_home=is_home,
                            **{stat.lower(): number(row.get(stat)) for stat in base_stats + Q1_STATS}, **extra))
    return dict(player=player_name, games=records)


@router.get('/player/{player_name}/chart-data')
def get_player_chart_data(player_name: str, stat: str = 'PTS', games: int = Query(20, ge=5, le=200),
                          league: League = 'nba', direction: Literal['over', 'under'] = 'over',
                          line: float | None = Query(None, allow_inf_nan=False)):
    stat = stat.upper()
    if stat not in STATS:
        raise HTTPException(400, f"Unknown stat '{stat}'")
    whole = _player(player_name, league)
    frame = whole.tail(games)
    columns = stat.split('+')
    if any(col not in frame for col in columns):
        raise HTTPException(503, 'This statistic is unavailable in player history')
    values = frame[columns].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=len(columns))
    context = _context_fields(_game_context(whole, league), len(frame))
    records = []
    for (_, row), raw, extra in zip(frame.iterrows(), values, context):
        value = number(raw)
        hit = None if line is None or value is None else (value > line if direction == 'over' else value < line)
        matchup = str(row.get('MATCHUP') or '')
        is_home = True if ' vs. ' in matchup else False if ' @ ' in matchup else None
        if is_home is None and 'is_home' in frame:
            home_value = number(row.get('is_home'))
            if home_value in (0, 1):
                is_home = bool(home_value)
        season = row.get('SEASON')
        records.append(dict(date=row['_date'].date().isoformat(), opponent=matchup,
                            value=value, hit=hit, season=None if pd.isna(season) else str(season),
                            is_home=is_home, **extra))
    return dict(player=player_name, stat=stat, line=line, direction=direction,
                avg=number(values.mean()), l5_avg=number(values.tail(5).mean()), games=records)


@router.get('/player/{player_name}/stats')
def get_player_stats(player_name: str, league: League = 'nba'):
    frame = _player(player_name, league)
    if frame.empty:
        raise HTTPException(503, 'No completed player games available')
    latest = frame.iloc[-1]
    season = frame[frame['SEASON'].astype(str) == str(latest['SEASON'])] if 'SEASON' in frame else frame
    columns = [col for col in ['PTS', 'AST', 'REB', 'FG3M', 'STL', 'BLK'] if col in frame]
    def averages(rows):
        return {col: number(pd.to_numeric(rows[col], errors='coerce').mean()) for col in columns}
    fg_pct = None
    if 'FG_PCT' in season:
        season_fg = number(pd.to_numeric(season['FG_PCT'], errors='coerce').mean())
        if season_fg is not None:
            fg_pct = round(season_fg * 100, 1)
    injury_status, injury_reason = _injury_context(player_name, league)
    position = ''
    position_path = get_config(league).data_dir / 'player_positions.csv'
    if position_path.exists():
        positions = pd.read_csv(position_path)
        matches = positions[positions['PLAYER_NAME'].str.casefold() == player_name.casefold()]
        if not matches.empty:
            position = str(matches.iloc[-1].get('POSITION', ''))
    projections = {}
    when = datetime.now(ZoneInfo('America/New_York')).date()
    for stat in ('PTS', 'REB', 'AST'):
        try:
            model = predictor(league, stat)
            if model is not None:
                result = model.predict_player_game(player_name, frame, game_date=when)
                value = number(result.get(f'predicted_{stat.lower()}'))
                if value is not None:
                    projections[stat] = round(value, 1)
        except Exception:
            logger.exception('Player projection unavailable: %s %s %s', league, player_name, stat)
    teams = frame['TEAM_ABBREVIATION'].dropna().astype(str).str.strip() if 'TEAM_ABBREVIATION' in frame else []
    teams = [team for team in teams if team and team.lower() not in ('nan', 'none')]
    return dict(player=player_name, team=teams[-1] if teams else '', position=position,
                headshot_url=_headshot_url(frame, league), fg_pct=fg_pct,
                injury_status=injury_status, injury_reason=injury_reason,
                season_avgs=averages(season), l5_avgs=averages(frame.tail(5)), games_played=len(season),
                history_through=latest['_date'].date().isoformat(),
                history_age_days=(pd.Timestamp(when) - latest['_date'].normalize()).days,
                projections=projections, projection_context=None,
                projection_message=('Model estimates using recent history; next opponent and venue are not set.'
                                    if projections else 'Trained player models are unavailable.'))
