"""WNBA matchup hit-rate board from the same computation as the Dash view."""
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

from fastapi import APIRouter

from api.data import history

router = APIRouter(tags=['hitrates'])
logger = logging.getLogger(__name__)


@router.get('/wnba/hitrates')
def get_wnba_hitrates():
    from utils.wnba_data_fetch import get_todays_wnba_games
    from utils.wnba_hit_rates import compute_hit_rates

    target_date = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    try:
        schedule = get_todays_wnba_games(target_date)
    except Exception:
        logger.exception('WNBA schedule unavailable for hit rates')
        return dict(target_date=target_date, games=[], message='WNBA schedule is unavailable.')
    if not schedule:
        return dict(target_date=target_date, games=[], message='No WNBA games scheduled today.')

    groups = compute_hit_rates(history('wnba'), schedule, n_games=10, min_hits=8, min_avg_min=15.0)
    return dict(target_date=target_date,
                games=[dict(matchup=group['matchup'], home=group['home'], away=group['away'],
                            entries=[asdict(entry) for entry in group['entries'][:12]],
                            total_count=len(group['entries'])) for group in groups],
                message=None if groups else 'No players met the hit-rate threshold on this slate.')
