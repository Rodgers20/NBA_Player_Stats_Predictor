"""Free ESPN game-market lines for the NBA and WNBA.

The public ESPN feeds are best-effort. Missing markets stay missing, and this
module never falls back to The Odds API or consumes its credit budget.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
import math
import re
import time
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

_TTL = 15 * 60
_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/{league}/scoreboard'
_CORE = ('https://sports.core.api.espn.com/v2/sports/basketball/leagues/'
         '{league}/events/{event_id}/competitions/{competition_id}/odds')
_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; BasketballResearch/1.0)'}
_EXCLUDE_PROVIDERS = ('numberfire', 'accuscore', 'teamrankings')
_ABBR = {'NY': 'NYK', 'GS': 'GSW', 'SA': 'SAS', 'NO': 'NOP', 'UTAH': 'UTA',
         'NJ': 'BKN', 'WSH': 'WAS', 'PHO': 'PHX', 'LAC': 'LAC',
         'LV': 'LVA', 'LA': 'LAS'}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _team_abbr(competitor, league):
    team = competitor.get('team') or {}
    raw = str(team.get('abbreviation') or '').upper()
    if league == 'wnba' and raw in ('GS', 'GSW'):
        return 'GSV'
    return _ABBR.get(raw, raw)


def _price(item, side):
    team = item.get(f'{side}TeamOdds') or {}
    value = team.get('moneyLine')
    if value is None:
        value = item.get(f'{side}MoneyLine')
    return _number(value)


def _home_spread(item, home, away):
    nested = (item.get('homeTeamOdds') or {}).get('current') or {}
    point = nested.get('pointSpread') or {}
    for value in (point.get('american'), point.get('alternateDisplayValue')):
        line = _number(value)
        if line is not None:
            return line
    spread = item.get('spread')
    if isinstance(spread, dict):
        return _number(((spread.get('home') or {}).get('line')))
    line = _number(spread)
    if line is None:
        return None
    details = str(item.get('details') or '').upper().strip()
    # ESPN usually reports the home line, but older feeds can report the
    # favorite's line. Prefer an explicit team tag whenever one is present.
    match = re.match(r'^([A-Z]{2,4})\s+([+-]?\d+(?:\.\d+)?)$', details)
    if match:
        team, displayed = match.groups()
        marked = _number(displayed)
        if team == away and marked is not None:
            return -marked
        if team == home and marked is not None:
            return marked
    return line


def _parse_items(items, home, away, fetched_at):
    options = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        provider = item.get('provider') or {}
        name = str(provider.get('name') or '').strip()
        if not name or any(blocked in name.lower() for blocked in _EXCLUDE_PROVIDERS):
            continue
        spread = _home_spread(item, home, away)
        total = _number(item.get('overUnder'))
        home_ml, away_ml = _price(item, 'home'), _price(item, 'away')
        completeness = sum(value is not None for value in (spread, total, home_ml, away_ml))
        if not completeness:
            continue
        priority = _number(provider.get('priority')) or 0
        options.append((completeness, priority, name, {
            'spread': {'home_line': spread} if spread is not None else None,
            'total': {'line': total} if total is not None else None,
            'h2h': {'home_price': home_ml, 'away_price': away_ml}
                   if home_ml is not None or away_ml is not None else None,
            'bookmaker': name,
            'source': 'ESPN',
            'fetched_at': fetched_at,
        }))
    if not options:
        return None
    return max(options, key=lambda option: (option[0], option[1]))[3]


def _fetch_core(league, event_id, competition_id, home, away, fetched_at):
    try:
        url = _CORE.format(league=league, event_id=event_id, competition_id=competition_id)
        response = requests.get(url, headers=_HEADERS, timeout=4)
        response.raise_for_status()
        return _parse_items(response.json().get('items'), home, away, fetched_at)
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.info('ESPN %s odds unavailable for %s: %s', league, event_id, exc)
        return None


def get_game_odds(league='nba', target_date=None, events=None, force_refresh=False):
    """Return ``AWAY@HOME`` quotes, cached per league and Eastern slate date.

    ``events`` accepts an ESPN scoreboard event list for callers that already
    have one. Otherwise one free scoreboard request discovers ESPN game IDs.
    """
    if league not in ('nba', 'wnba'):
        raise ValueError('Unsupported league')
    date = target_date or datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    key = (league, date)
    now = time.time()
    cached = _CACHE.get(key)
    if not force_refresh and cached and 0 <= now - cached[0] < _TTL:
        return cached[1].copy()

    if events is None:
        try:
            response = requests.get(_SCOREBOARD.format(league=league),
                                    params={'dates': date.replace('-', '')},
                                    headers=_HEADERS, timeout=5)
            response.raise_for_status()
            events = response.json().get('events') or []
        except (requests.RequestException, ValueError, TypeError) as exc:
            logger.warning('ESPN %s scoreboard odds unavailable: %s', league, exc)
            return cached[1].copy() if cached and 0 <= now - cached[0] < _TTL else {}

    fetched_at = datetime.now(timezone.utc).isoformat()
    quotes = {}
    missing = []
    for event in events:
        competitions = event.get('competitions') or []
        if not competitions:
            continue
        comp = competitions[0]
        home = next((_team_abbr(c, league) for c in comp.get('competitors') or []
                     if c.get('homeAway') == 'home'), '')
        away = next((_team_abbr(c, league) for c in comp.get('competitors') or []
                     if c.get('homeAway') == 'away'), '')
        if not home or not away:
            continue
        matchup = f'{away}@{home}'
        quote = _parse_items(comp.get('odds'), home, away, fetched_at)
        if quote:
            quotes[matchup] = quote
        if not quote or not all((quote.get('spread'), quote.get('total'),
                                 (quote.get('h2h') or {}).get('home_price'),
                                 (quote.get('h2h') or {}).get('away_price'))):
            event_id = event.get('id')
            competition_id = comp.get('id') or event_id
            if event_id and competition_id:
                missing.append((matchup, str(event_id), str(competition_id), home, away))

    if missing:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_fetch_core, league, eid, cid, home, away, fetched_at): matchup
                       for matchup, eid, cid, home, away in missing}
            for future in as_completed(futures):
                quote = future.result()
                if quote and (not quotes.get(futures[future]) or
                              _completeness(quote) > _completeness(quotes[futures[future]])):
                    quotes[futures[future]] = quote

    _CACHE[key] = (time.time(), quotes)
    return quotes.copy()


def _completeness(quote):
    return sum(value is not None for value in (
        (quote.get('spread') or {}).get('home_line'),
        (quote.get('total') or {}).get('line'),
        (quote.get('h2h') or {}).get('home_price'),
        (quote.get('h2h') or {}).get('away_price')))
