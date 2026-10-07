"""Schedule and model endpoints. Odds are read from existing memory only."""
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.data import League, history, number
from utils.league_config import get_config

router = APIRouter(tags=['games'])
logger = logging.getLogger(__name__)


def _schedule(league):
    try:
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
def get_games(league: League = 'nba'):
    records, target = _schedule(league)
    odds = _cached_odds(league, target) if records else {}
    games = []
    for row in records:
        home, away = row.get('HOME_TEAM', ''), row.get('AWAY_TEAM', '')
        quote = odds.get(f'{away}@{home}', {})
        games.append(dict(matchup=f'{away} @ {home}', home_team=home, away_team=away,
                          game_time=row.get('GAME_TIME', ''), game_id=row.get('GAME_ID'),
                          status_text=row.get('GAME_STATUS_TEXT'),
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
def get_predictions(league: League = 'nba'):
    records, target = _schedule(league)
    if not records:
        return dict(target_date=target, predictions=[], errors=[])
    # Dash uses this GamePredictor with league-specific team defense and logs
    # for both NBA and WNBA. Keep the same source of model estimates here.
    try:
        from utils.game_predictor import GamePredictor
        path = get_config(league).data_dir / 'team_defensive_stats.csv'
        model = GamePredictor(pd.read_csv(path), history(league))
    except Exception as exc:
        logger.exception('Game model unavailable')
        raise HTTPException(503, 'Game prediction data or model is unavailable') from exc
    odds = _cached_odds(league, target)
    predictions, errors = [], []
    for row in records:
        home, away = row.get('HOME_TEAM', ''), row.get('AWAY_TEAM', '')
        try:
            if league == 'nba':
                result = model.predict_game(home, away,
                                            home_injuries=_team_injuries(league, home),
                                            away_injuries=_team_injuries(league, away))
            else:
                result = model.predict_game(home, away)
            quote = odds.get(f'{away}@{home}', {})
            market_spread = number((quote.get('spread') or {}).get('home_line'))
            market_total = number((quote.get('total') or {}).get('line'))
            pick = model.get_pick(result, home=home, away=away,
                                  actual_spread=market_spread, actual_total=market_total) if hasattr(model, 'get_pick') else {}
            reasoning = result.get('reasoning') or {}
            predictions.append(dict(matchup=f'{away} @ {home}', home_team=home, away_team=away,
                                    predicted_winner=result['winner'], spread=number(result['predicted_spread']),
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
