"""Confirmed ESPN match lineups, joined to our fixture calendar by both clubs/date.

Provider IDs never double as FUTBALL player IDs. Scores are an explicitly limited
FUTBALL contribution index, not a provider rating or a prediction.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import logging
import threading

import httpx

from . import clubs, config
from .sources import _cache, football_data_org as fdo

LEAGUES = {'E0': 'eng.1', 'SP1': 'esp.1', 'UCL': 'uefa.champions'}
BASE = 'https://site.api.espn.com/apis/site/v2/sports/soccer'
RAW = config.RAW / 'espn'
_lock = threading.Lock()
log = logging.getLogger(__name__)


def fetch(code: str, endpoint: str, params: dict) -> dict:
    key = hashlib.sha256(repr(sorted(params.items())).encode()).hexdigest()[:20]
    path = RAW / code / f'{endpoint}-{key}.json'
    cached = _cache.read_json(path)
    ttl = 86400 if endpoint == 'summary' and cached and cached.get('rosters') else 900
    if cached and _cache.is_fresh(path, ttl):
        return cached
    with _lock:
        if _cache.is_fresh(path, ttl):
            return _cache.read_json(path) or {}
        try:
            response = httpx.get(f'{BASE}/{LEAGUES[code]}/{endpoint}', params=params, timeout=20)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError('Unexpected ESPN payload')
            _cache.write_json(path, body)
            return body
        except (httpx.HTTPError, ValueError, OSError):
            log.warning('ESPN %s %s unavailable', code, endpoint)
            return {**cached, '_stale': True} if cached else {}


def canonical(name: str) -> str:
    entry = clubs.find(name)
    return entry['name'] if entry else name.casefold().strip()


def fixtures(code: str, year: int | None = None) -> list[dict]:
    payload = fdo.read_matches(code) or {}
    if fdo.available():
        try:
            payload = fdo.fetch_matches(code)
        except Exception:
            log.warning('Fixture refresh unavailable for %s; using cached calendar', code)
    rows = payload.get('matches') or []
    if year is not None:
        rows = [r for r in rows if str((r.get('season') or {}).get('startDate', ''))[:4] == str(year)]
    return rows


def finished(row: dict) -> bool:
    return row.get('status') == 'FINISHED' and (row.get('utcDate') or '') <= datetime.now(timezone.utc).isoformat()


def match_summary(row: dict, code: str) -> dict:
    day = date.fromisoformat(row['utcDate'][:10])
    dates = f'{(day-timedelta(days=1)):%Y%m%d}-{(day+timedelta(days=1)):%Y%m%d}'
    board = fetch(code, 'scoreboard', {'dates': dates, 'limit': 100})
    wanted = {side: canonical(row[f'{side}Team']['name']) for side in ('home', 'away')}
    for event in board.get('events') or []:
        if event.get('date', '')[:10] != row['utcDate'][:10]:
            continue
        competition = (event.get('competitions') or [{}])[0]
        actual = {c['homeAway']: canonical(c['team']['displayName']) for c in competition.get('competitors', [])}
        if actual != wanted or not (event.get('status', {}).get('type', {}).get('completed')):
            continue
        body = fetch(code, 'summary', {'event': event['id']})
        header = body.get('header') or {}
        if str(header.get('id')) != str(event['id']):
            return {}
        return {**body, '_source_url': f'https://www.espn.com/soccer/match/_/gameId/{event["id"]}', '_stale': body.get('_stale', False) or board.get('_stale', False)}
    return {}


# ESPN's position vocabulary. Unknown roles are not guessed into a line.
GROUPS = {
    'G': 'GK', 'GK': 'GK', 'D': 'DEF', 'CD': 'DEF', 'CD-L': 'DEF', 'CD-R': 'DEF',
    'LB': 'DEF', 'RB': 'DEF', 'LWB': 'DEF', 'RWB': 'DEF',
    'M': 'MID', 'CM': 'MID', 'CM-L': 'MID', 'CM-R': 'MID', 'DM': 'MID',
    'DM-L': 'MID', 'DM-R': 'MID', 'LM': 'MID', 'RM': 'MID', 'AM': 'MID',
    'AM-L': 'MID', 'AM-R': 'MID', 'F': 'ATT', 'CF': 'ATT', 'CF-L': 'ATT',
    'CF-R': 'ATT', 'LF': 'ATT', 'RF': 'ATT', 'LW': 'ATT', 'RW': 'ATT', 'SS': 'ATT',
}


def starters(roster: dict) -> list[dict]:
    result = []
    seen = set()
    for row in roster.get('roster') or []:
        athlete = row.get('athlete') or {}
        pid = athlete.get('id')
        if row.get('starter') is not True or not pid or pid in seen:
            continue
        seen.add(pid)
        stats = {s['name']: s.get('value') for s in row.get('stats', [])}
        position = row.get('position') or {}
        result.append({'provider_id': str(pid), 'player_id': None,
            'name': athlete.get('displayName') or athlete.get('fullName'),
            'short_name': athlete.get('lastName') or athlete.get('displayName'),
            'shirt': row.get('jersey'), 'position': position.get('displayName'),
            'group': GROUPS.get(position.get('abbreviation')), 'stats': stats,
            'goals': stats.get('totalGoals'), 'assists': stats.get('goalAssists'),
            'rating': None, 'played': 1,
            'source_url': next((x['href'] for x in athlete.get('links', []) if 'playercard' in x.get('rel', [])), None)})
    return result


def pitch_slots(people: list[dict], *, detailed: bool = False) -> list[dict]:
    # Spread players across their reported positional lines; no invented slots.
    slots = []
    bands = [('ATT', .13), ('AM', .32), ('MID', .52), ('DEF', .73), ('GK', .91)] if detailed else [('ATT', .16), ('MID', .40), ('DEF', .67), ('GK', .89)]
    for group, y in bands + [(None, .53)]:
        line = [p for p in people if ('AM' if detailed and 'Attacking Midfielder' in (p['position'] or '') else p['group']) == group]
        line.sort(key=lambda p: (0 if 'Left' in (p['position'] or '') else 2 if 'Right' in (p['position'] or '') else 1, p['provider_id']))
        for i, person in enumerate(line):
            slots.append({'slot': person['provider_id'], 'line': group or 'Unknown',
                'abbreviation': group or '?', 'x': (i + .5) / len(line), 'y': y, 'player': person})
    return slots


def latest(team: str, code: str) -> dict:
    empty = {'team': team, 'competition_code': code, 'competition': '', 'slots': [], 'filled': 0,
             'note': 'The latest match lineup is not available from the connected sources.', 'source': 'ESPN'}
    target = canonical(team)
    matches = [(r, c) for c in LEAGUES for r in fixtures(c) if finished(r)
               and target in [canonical(r[s]['name']) for s in ('homeTeam', 'awayTeam')]]
    if not matches:
        return empty
    row, match_code = max(matches, key=lambda pair: pair[0]['utcDate'])
    body = match_summary(row, match_code)
    roster = next((r for r in body.get('rosters', []) if canonical(r.get('team', {}).get('displayName', '')) == target), {})
    people = starters(roster)
    base = {**empty, 'competition_code': match_code, 'competition': config.LEAGUES.get(match_code, {}).get('name', match_code),
        'match_date': row['utcDate'], 'match_label': f'{row["homeTeam"]["name"]} vs {row["awayTeam"]["name"]}',
        'formation': roster.get('formation'), 'source_url': body.get('_source_url')}
    if len(people) != 11:
        return {**base, 'note': 'A complete confirmed starting XI has not been published for this match.'}
    from .lineup_season import enrich
    season_stats = enrich(people, team, code)
    return {**base, **season_stats, 'filled': 11, 'slots': pitch_slots(people, detailed=True),
        'note': 'Latest confirmed starters · ESPN. Badges show season totals for the selected competition.' + (' Cached data; refresh unavailable.' if body.get('_stale') else '')}


def contribution_score(player: dict, conceded: int | None) -> float | None:
    s = player['stats']
    required = ('totalGoals', 'goalAssists', 'shotsOnTarget', 'yellowCards', 'redCards', 'ownGoals')
    if any(not isinstance(s.get(k), (int, float)) for k in required):
        return None
    value = 6 + 1.1*s['totalGoals'] + .75*s['goalAssists'] + .1*s['shotsOnTarget'] - .2*s['yellowCards'] - 1.2*s['redCards'] - .8*s['ownGoals']
    if player['group'] in ('GK', 'DEF') and conceded == 0:
        value += .6  # Team clean sheet, not an inferred individual clean sheet.
    if player['group'] == 'GK':
        if not isinstance(s.get('saves'), (int, float)):
            return None
        value += .15*s['saves'] - .15*(s.get('goalsConceded') or 0)
    return round(max(1, min(10, value)), 2)


def team_of_week(code: str, week: int | None = None, year: int | None = None) -> dict:
    empty = {'competition_code': code, 'competition': config.LEAGUES.get(code, {}).get('name', code),
             'match_week': week, 'rounds': [], 'players': [], 'note': 'Match statistics are unavailable for this competition and season.'}
    if code not in LEAGUES:
        return empty
    rows = fixtures(code, year)
    rounds = sorted({r['matchday'] for r in rows if finished(r) and isinstance(r.get('matchday'), int)})
    empty['rounds'] = rounds
    if not rounds:
        return empty
    if week is None:
        complete = [n for n in rounds if all(finished(r) for r in rows if r.get('matchday') == n)]
        week = (complete or rounds)[-1]
    selected = [r for r in rows if r.get('matchday') == week]
    completed = [r for r in selected if finished(r)]
    empty['match_week'] = week
    if not completed:
        return empty
    with ThreadPoolExecutor(max_workers=3) as pool:
        summaries = list(pool.map(lambda r: match_summary(r, code), completed))
    candidates = []
    covered = 0
    for row, body in zip(completed, summaries):
        rosters = body.get('rosters') or []
        if len(rosters) != 2 or any(len(starters(r)) != 11 for r in rosters):
            continue
        covered += 1
        for roster in rosters:
            opponent = 'away' if roster.get('homeAway') == 'home' else 'home'
            conceded = (row.get('score', {}).get('fullTime') or {}).get(opponent)
            for p in starters(roster):
                score = contribution_score(p, conceded)
                if score is None or p['group'] is None:
                    continue
                candidates.append({**p, 'rating': score, 'team': roster['team']['displayName'],
                    'club': canonical(roster['team']['displayName']), 'minutes': None,
                    'source_url': body.get('_source_url')})
    candidates.sort(key=lambda p: (-p['rating'], p['provider_id']))
    # One appearance per player, even when a postponed match shares a round.
    unique = {}
    for p in candidates:
        unique.setdefault(p['provider_id'], p)
    groups = {g: [p for p in unique.values() if p['group'] == g] for g in ('GK', 'DEF', 'MID', 'ATT')}
    choices = []
    for d, m, a in [(4,3,3), (4,4,2), (4,5,1), (3,4,3), (3,5,2), (5,3,2), (5,4,1)]:
        counts = {'GK':1,'DEF':d,'MID':m,'ATT':a}
        if all(len(groups[g]) >= n for g,n in counts.items()):
            picks = [p for g,n in counts.items() for p in groups[g][:n]]
            choices.append((sum(p['rating'] for p in picks), f'{d}-{m}-{a}', picks))
    if not choices:
        return {**empty, 'note': f'Insufficient verified positional coverage for an XI ({covered}/{len(selected)} matches available).'}
    _, formation, picks = max(choices, key=lambda x:x[0])
    return {**empty, 'players': pitch_slots(picks), 'formation': formation,
        'note': f'FUTBALL contribution score · Round {week} · {covered}/{len(selected)} matches covered. Confirmed starters only. Goals, assists, shots on target, cards, saves and team clean sheets; passing and defensive actions unavailable. Not a provider rating.' + (' Provisional round.' if covered < len(selected) else '') + (' Cached match data.' if any(b.get('_stale') for b in summaries) else ''),
        'source': 'ESPN', 'season_year': year}
