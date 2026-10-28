"""Spend the shared Odds API credits on whichever league actually has games.

Each prop type costs one credit per game; allocate() spreads the day's credits so every game is
covered before any game gets deeper markets. When a league is off-season it has no events and its
share flows to the league that is playing.
"""
import asyncio
import calendar
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
from utils import odds_budget
from utils.odds_fetcher import upcoming_events
from utils.slate import next_slate

logger = logging.getLogger(__name__)
ET = ZoneInfo('America/New_York')
# Prop types bought per game. NBA buys its markets in priority order (utils.odds_fetcher.MARKET_PRIORITY),
# so 2 means points and rebounds; WNBA always buys its three. Raise MIN_MARKETS after upgrading the plan.
MIN_MARKETS = {'nba': 2, 'wnba': 3}
MAX_MARKETS = {'nba': 7, 'wnba': 3}
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


def upcoming(league, target=None):
    """Unstarted events on one Eastern date; with no `target`, the next date (within
    LOOKAHEAD_DAYS) that has any. The /events endpoint costs no credits."""
    module = _module(league)
    if not module.API_KEY:
        return []
    response = odds_budget.request(requests.get, f'{module.BASE_URL}/sports/{module.SPORT}/events',
                                   {'apiKey': module.API_KEY, 'dateFormat': 'iso'})
    if response is None:
        return []
    events = response.json()
    if target is not None:
        return upcoming_events(events, target)
    return next_slate(lambda day: upcoming_events(events, day))[1]


def event_date(event):
    return datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00')).astimezone(ET).date().isoformat()


def allocate(counts, credits):
    """{league: (games to price, prop types per game)}. Each prop type costs one credit per game.

    Coverage first: every game gets the minimum number of prop types before any game gets more, so a
    full slate is never cut to two games; on a light night the spare credits buy deeper markets.
    Leagues with games share the credits in proportion to their slates; an idle league's share flows on.
    """
    total = sum(counts.values())
    plan = {}
    for league, games in counts.items():
        share = credits * games // total if total else 0
        least, most = MIN_MARKETS[league], MAX_MARKETS[league]
        if not games or share < least:
            plan[league] = (0, 0)
        elif share // games >= least:
            plan[league] = (games, min(most, share // games))
        else:  # not enough for every game: as many games as possible, earliest tips first
            plan[league] = (share // least, least)
    return plan


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
    """{league: (games to price, prop types per game)}. `events` is injectable for tests."""
    target = today(now)
    events = events if events is not None else {league: upcoming(league, target) for league in LEAGUES}
    return allocate({league: len(found) for league, found in events.items()}, headroom(kind, now))


def refresh(league, events, target_date=None, markets=None):
    """Buy quotes for `events` games (`markets` prop types each); returns the quote dict (empty if nothing was bought)."""
    if not events:
        return {}
    module = _module(league)
    if league == 'wnba':
        return module.get_live_wnba_odds(force_refresh=True, target_date=target_date, max_events=events)
    return module.get_live_odds(force_refresh=True, target_date=target_date, max_events=events,
                                **({'markets': markets} if markets else {}))


def manual_refresh(league, now=None):
    """Snapshot or button press: spend today's allowance on this league's next slate, even the day before.
    Today's slate waits for MORNING_HOUR: before it, books have rarely posted player props, and the first
    purchase of the day now covers every game, so buying early could waste the whole day's credits."""
    now = now or datetime.now(timezone.utc)
    found = {name: upcoming(name) for name in LEAGUES}
    target = event_date(found[league][0]) if found[league] else None
    if target == today(now) and now.astimezone(ET).hour < MORNING_HOUR:
        return {}
    games, markets = plan('manual', found, now).get(league, (0, 0))
    return refresh(league, games, target, markets)


def q1_refresh(now=None):
    """First-quarter markets for today's NBA slate (its first games; see odds_fetcher.get_q1_odds), after MORNING_HOUR.
    Never for tomorrow's slate: books post 1Q lines late, and an early look would spend one of each game's two tries."""
    now = now or datetime.now(timezone.utc)
    found = upcoming('nba')
    if not found or event_date(found[0]) != today(now) or now.astimezone(ET).hour < MORNING_HOUR:
        return {}
    from utils import odds_fetcher
    return odds_fetcher.get_q1_odds(today(now))


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
    for league, (count, markets) in plan(kind, events, now).items():
        if count:
            refresh(league, count, markets=markets)
            refresh_props(league=league, fetch_odds=False)
    return kind


async def run_forever():
    while True:
        try:
            await asyncio.to_thread(tick)
        except Exception:
            logger.exception('Scheduled odds refresh failed')
        await asyncio.sleep(TICK_SECONDS)
