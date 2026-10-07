"""Spend the shared Odds API credits on whichever league actually has games.

The models only price PTS/REB/AST, so every event costs CORE_COST credits. When a league
is off-season it has no events and its share flows to the league that is playing.
"""
import asyncio
import calendar
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
from utils import odds_budget
from utils.odds_fetcher import upcoming_events

logger = logging.getLogger(__name__)
ET = ZoneInfo('America/New_York')
CORE_COST = 3          # player_points + player_rebounds + player_assists
RESERVE = 25           # never plan the provider account below this
LEAGUES = ('nba', 'wnba')
MORNING_HOUR = 9
PRETIP_WINDOW = timedelta(hours=2)
TICK_SECONDS = 15 * 60


def _module(league):
    if league == 'wnba':
        from utils import wnba_odds_fetcher as module
    else:
        from utils import odds_fetcher as module
    return module


def today(now=None):
    return (now or datetime.now(timezone.utc)).astimezone(ET).date().isoformat()


def upcoming(league, target):
    """Unstarted events on the Eastern date. The /events endpoint costs no credits."""
    module = _module(league)
    if not module.API_KEY:
        return []
    response = odds_budget.request(requests.get, f'{module.BASE_URL}/sports/{module.SPORT}/events',
                                   {'apiKey': module.API_KEY, 'dateFormat': 'iso'})
    return upcoming_events(response.json(), target) if response is not None else []


def split_events(counts, credits):
    """Share credits fairly across leagues with games; a league's unused share goes to the others."""
    left = credits // CORE_COST
    alloc = dict.fromkeys(counts, 0)
    active = [league for league, n in counts.items() if n]
    while left and active:
        share = max(1, left // len(active))
        for league in list(active):
            take = min(share, counts[league] - alloc[league], left)
            alloc[league] += take
            left -= take
            if alloc[league] >= counts[league]:
                active.remove(league)
            if not left:
                break
    return alloc


def headroom(kind, now=None):
    """Credits this run may spend: today's fair share of what is left this month."""
    now = (now or datetime.now(timezone.utc)).astimezone(ET)
    state = odds_budget.status()
    usable = odds_budget.MONTHLY_LIMIT - state['monthly']
    if state['remaining'] is not None:
        usable = min(usable, state['remaining'] - RESERVE)
    days_left = calendar.monthrange(now.year, now.month)[1] - now.day + 1
    day = min(odds_budget.DAILY_LIMIT, -(-max(usable, 0) // days_left))
    room = max(0, min(day - state['daily'], usable))
    return -(-room // 2) if kind == 'morning' else room


def plan(kind='manual', events=None, now=None):
    """{league: number of events to price}. `events` is injectable for tests."""
    target = today(now)
    events = events if events is not None else {league: upcoming(league, target) for league in LEAGUES}
    return split_events({league: len(found) for league, found in events.items()}, headroom(kind, now))


def refresh(league, events):
    """Buy quotes for `events` games; returns the quote dict (empty if nothing was bought)."""
    if not events:
        return {}
    module = _module(league)
    fetch = module.get_live_wnba_odds if league == 'wnba' else module.get_live_odds
    return fetch(force_refresh=True, max_events=events)


# ── Scheduler ───────────────────────────────────────────────────────────────

def _runs(day):
    with odds_budget.connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS odds_runs (day TEXT, kind TEXT, PRIMARY KEY (day, kind))')
        return {kind for (kind,) in db.execute('SELECT kind FROM odds_runs WHERE day=?', (day,))}


def _mark(day, kind):
    with odds_budget.connection() as db:
        db.execute('INSERT OR IGNORE INTO odds_runs VALUES (?,?)', (day, kind))


def due_kind(now, first_tip, done):
    """Which scheduled refresh (if any) should run now."""
    if 'morning' not in done and now.astimezone(ET).hour >= MORNING_HOUR:
        return 'morning'
    if 'pretip' not in done and first_tip is not None and timedelta(0) < first_tip - now <= PRETIP_WINDOW:
        return 'pretip'
    return None


def tick(now=None):
    """One scheduler pass; returns the kind it ran, or None. Evaluates props after buying quotes."""
    now = now or datetime.now(timezone.utc)
    target = today(now)
    done = _runs(target)
    events = {league: upcoming(league, target) for league in LEAGUES}
    starts = [datetime.fromisoformat(e['commence_time'].replace('Z', '+00:00')) for found in events.values() for e in found]
    kind = due_kind(now, min(starts) if starts else None, done)
    if kind is None:
        return None
    _mark(target, kind)
    from api.routes.props import refresh_props
    for league, count in plan(kind, events, now).items():
        if count:
            refresh(league, count)
            refresh_props(league=league, fetch_odds=False)
    return kind


async def run_forever():
    while True:
        try:
            await asyncio.to_thread(tick)
        except Exception:
            logger.exception('Scheduled odds refresh failed')
        await asyncio.sleep(TICK_SECONDS)
