"""Schedule and model endpoints. Odds are read from existing memory only."""
from datetime import date, datetime
from zoneinfo import ZoneInfo
import logging
import re
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.data import League, history, number
from utils.league_config import get_config

router = APIRouter(tags=['games'])
logger = logging.getLogger(__name__)


def _validated_date(value):
    if value is None:
        return None
    try:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError('Expected YYYY-MM-DD')
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, 'game_date must be a valid YYYY-MM-DD date') from exc


def _espn_schedule(league, game_date):
    """Fetch exactly the requested date, including completed and live games."""
    from utils.basketball_content import _fetch
    from utils.espn_game_odds import _team_abbr
    rows = []
    board = _fetch(league, 'scoreboard', dates=game_date.replace('-', ''))
    for event in board.get('events') or []:
        competitions = event.get('competitions') or []
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get('competitors') or []
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        if not home or not away:
            continue
        status = (comp.get('status') or event.get('status') or {}).get('type') or {}
        state = status.get('state') or ('post' if status.get('completed') else 'pre')
        row = dict(GAME_ID=event.get('id'), GAME_TIME=event.get('date'),
                   START_TIME=event.get('date'), GAME_STATUS_STATE=state,
                   GAME_STATUS_TEXT=status.get('shortDetail') or status.get('description'),
                   VENUE=(comp.get('venue') or {}).get('fullName'),
                   NEUTRAL_SITE=bool(comp.get('neutralSite', False)),
                   SEASON_TYPE=(event.get('season') or board.get('season') or {}).get('type'),
                   BROADCAST=', '.join(name for b in comp.get('broadcasts') or []
                                       for name in b.get('names') or []))
        for side, competitor in (('HOME', home), ('AWAY', away)):
            team = competitor.get('team') or {}
            overall = next((r.get('summary') for r in competitor.get('records') or []
                            if r.get('type') == 'total' or r.get('name') == 'overall'), '') or ''
            record = overall.split('-')
            row.update({f'{side}_TEAM': _team_abbr(competitor, league),
                        f'{side}_NAME': team.get('displayName'),
                        f'{side}_LOGO': team.get('logo'),
                        f'{side}_COLOR': team.get('color'),
                        f'{side}_WINS': record[0] if len(record) == 2 else None,
                        f'{side}_LOSSES': record[1] if len(record) == 2 else None,
                        f'{side}_SCORE': competitor.get('score') if state in ('in', 'post') else None})
        rows.append(row)
    return rows, game_date


def _schedule(league, game_date=None):
    try:
        if game_date:
            return _espn_schedule(league, game_date)
        if league == 'wnba':
            from utils.wnba_data_fetch import get_todays_wnba_games
            from utils.slate import next_slate
            target, result = next_slate(get_todays_wnba_games)
            return [dict(HOME_TEAM=g['home']['abbrev'], AWAY_TEAM=g['away']['abbrev'],
                         GAME_TIME=g.get('tip_time_et', ''), GAME_ID=g.get('game_id'),
                         GAME_STATUS_TEXT=g.get('status_text') or g.get('status'),
                         HOME_NAME=g['home'].get('name'), AWAY_NAME=g['away'].get('name'),
                         HOME_WINS=g['home'].get('wins'), HOME_LOSSES=g['home'].get('losses'),
                         AWAY_WINS=g['away'].get('wins'), AWAY_LOSSES=g['away'].get('losses'),
                         HOME_SCORE=g['home'].get('score'), AWAY_SCORE=g['away'].get('score'))
                    for g in result], target
        from utils.data_fetch import get_upcoming_games
        frame, target = get_upcoming_games()
        if frame is None:
            raise ValueError('Schedule provider unavailable')
        return frame.to_dict('records'), str(target) if target else None
    except Exception as exc:
        logger.exception('Schedule unavailable for %s', league)
        raise HTTPException(503, 'Schedule provider is unavailable; try again later') from exc


def _status_state(text):
    value = str(text or '').lower()
    if 'final' in value or 'completed' in value:
        return 'post'
    if any(word in value for word in ('quarter', 'halftime', 'q1', 'q2', 'q3', 'q4', 'in progress')):
        return 'in'
    return 'pre'


def _cached_odds(league, target_date):
    from utils.espn_game_odds import get_game_odds
    return get_game_odds(league, target_date)


def _team_injuries(league: League, team: str) -> list[dict]:
    """Read the injury feeds used by the original Dash Games views."""
    try:
        if league == 'wnba':
            from utils.wnba_injuries import get_wnba_team_injuries
            entries = get_wnba_team_injuries(team)
        else:
            from utils.injury_news import get_team_injuries
            entries = get_team_injuries(team)
        return [dict(name=str(item.get('name') or item.get('player_name') or ''),
                     status=str(item.get('status') or ''),
                     reason=str(item.get('reason') or ''))
                for item in entries or [] if item.get('name') or item.get('player_name')]
    except Exception:
        logger.exception('Team injuries unavailable: %s %s', league, team)
        return []


@router.post('/games/refresh-lines')
def refresh_game_lines(league: League = 'nba'):
    """Refresh free ESPN basketball lines without touching paid odds credits."""
    from utils.espn_game_odds import get_game_odds
    from utils.slate import slate_date
    target = slate_date(league)
    quotes = get_game_odds(league, target, force_refresh=True)
    count = len(quotes)
    return dict(count=count, status='ready' if count else 'unavailable', source='ESPN',
                message=f'{count} ESPN game lines refreshed without odds API credits.' if count
                        else 'ESPN has no game lines available for this slate.')


@router.get('/games')
def get_games(league: League = 'nba', game_date: str | None = None):
    requested = _validated_date(game_date)
    records, target = _schedule(league, requested) if requested else _schedule(league)
    odds = _cached_odds(league, target) if records else {}
    games = []
    for row in records:
        home, away = row.get('HOME_TEAM', ''), row.get('AWAY_TEAM', '')
        quote = odds.get(f'{away}@{home}', {})
        games.append(dict(matchup=f'{away} @ {home}', home_team=home, away_team=away,
                          game_time=row.get('GAME_TIME', ''), game_id=row.get('GAME_ID'),
                          status_text=row.get('GAME_STATUS_TEXT'),
                          status_state=row.get('GAME_STATUS_STATE') or _status_state(row.get('GAME_STATUS_TEXT')),
                          start_time=row.get('START_TIME') or row.get('GAME_TIME'),
                          home_logo=row.get('HOME_LOGO'), away_logo=row.get('AWAY_LOGO'),
                          home_color=row.get('HOME_COLOR'), away_color=row.get('AWAY_COLOR'),
                          venue=row.get('VENUE'), broadcast=row.get('BROADCAST'),
                          home_name=row.get('HOME_NAME'), away_name=row.get('AWAY_NAME'),
                          home_wins=number(row.get('HOME_WINS')), home_losses=number(row.get('HOME_LOSSES')),
                          away_wins=number(row.get('AWAY_WINS')), away_losses=number(row.get('AWAY_LOSSES')),
                          home_score=number(row.get('HOME_SCORE')), away_score=number(row.get('AWAY_SCORE')),
                          home_injuries=_team_injuries(league, home), away_injuries=_team_injuries(league, away),
                          spread=number((quote.get('spread') or {}).get('home_line')),
                          total=number((quote.get('total') or {}).get('line')),
                          home_ml=number((quote.get('h2h') or {}).get('home_price')),
                          away_ml=number((quote.get('h2h') or {}).get('away_price')),
                          odds_source=quote.get('source'), odds_provider=quote.get('bookmaker'),
                          odds_updated_at=quote.get('fetched_at')))
    return dict(target_date=target, games=games, league=league)


@router.get('/games/predictions')
def get_predictions(league: League = 'nba', game_date: str | None = None):
    requested = _validated_date(game_date)
    if requested and requested < datetime.now(ZoneInfo('America/New_York')).date().isoformat():
        return dict(target_date=requested, predictions=[], errors=[],
                    message='No published historical forecast is available for this date.')
    records, target = _schedule(league, requested) if requested else _schedule(league)
    if not records:
        return dict(target_date=target, predictions=[], errors=[])
    # Dash uses this GamePredictor with league-specific team defense and logs
    # for both NBA and WNBA. Keep the same source of model estimates here.
    try:
        from utils.game_predictor import GamePredictor
        path = get_config(league).data_dir / 'team_defensive_stats.csv'
        model = GamePredictor(pd.read_csv(path), history(league), league=league, target_date=target)
    except Exception as exc:
        logger.exception('Game model unavailable')
        raise HTTPException(503, 'Game prediction data or model is unavailable') from exc
    odds = _cached_odds(league, target)
    predictions, errors = [], []
    for row in records:
        home, away = row.get('HOME_TEAM', ''), row.get('AWAY_TEAM', '')
        try:
            state = row.get('GAME_STATUS_STATE') or _status_state(row.get('GAME_STATUS_TEXT'))
            if state in ('in', 'post'):
                continue  # Current history is not a published pregame forecast.
            result = model.predict_game(home, away, target_date=target,
                                        neutral=bool(row.get('NEUTRAL_SITE', False)),
                                        venue=row.get('VENUE'),
                                        home_injuries=_team_injuries(league, home),
                                        away_injuries=_team_injuries(league, away))
            metadata = result.get('model_metadata')
            if metadata is not None:
                metadata['training_game_types'] = ['Regular Season', 'Playoffs']
                metadata['schedule_season_type'] = row.get('SEASON_TYPE')
                metadata['preseason_validated'] = False
                if row.get('SEASON_TYPE') == 1:
                    metadata['context_warning'] = 'Preseason rotations and minutes are outside model validation.'
                elif row.get('SEASON_TYPE') is None:
                    metadata['context_warning'] = 'Schedule game type unavailable; preseason performance is unvalidated.'
            quote = odds.get(f'{away}@{home}', {})
            market_spread = number((quote.get('spread') or {}).get('home_line'))
            market_total = number((quote.get('total') or {}).get('line'))
            pick = model.get_pick(result, home=home, away=away,
                                  actual_spread=market_spread, actual_total=market_total) if hasattr(model, 'get_pick') else {}
            reasoning = result.get('reasoning') or {}
            predictions.append(dict(matchup=f'{away} @ {home}', home_team=home, away_team=away,
                                    predicted_winner=result['winner'], spread=number(result['predicted_spread']),
                                    model_metadata=result.get('model_metadata'), data_quality=result.get('data_quality'),
                                    home_win_probability=result.get('home_win_probability'),
                                    winner_probability=result.get('winner_probability'),
                                    prediction_intervals=result.get('prediction_intervals'),
                                    total=number(result['predicted_total']), confidence=result['winner_confidence'],
                                    predicted_home_score=number(result.get('predicted_home_score')),
                                    predicted_away_score=number(result.get('predicted_away_score')),
                                    intel=[str(item) for item in result.get('intel', [])],
                                    winner_reason=reasoning.get('winner_reason'),
                                    spread_reason=reasoning.get('spread_reason'),
                                    total_reason=reasoning.get('total_reason'),
                                    market_spread=market_spread, market_total=market_total,
                                    spread_pick=pick.get('spread_pick'), spread_team=pick.get('spread_team'),
                                    spread_confidence=pick.get('spread_confidence'),
                                    total_pick=pick.get('total_pick'),
                                    total_confidence=pick.get('total_confidence')))
        except Exception:
            logger.exception('Prediction failed for %s @ %s', away, home)
            errors.append(dict(matchup=f'{away} @ {home}', message='Prediction unavailable'))
    return dict(target_date=target, predictions=predictions, errors=errors)
