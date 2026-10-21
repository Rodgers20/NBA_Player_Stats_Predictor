"""Free ESPN team schedules and actual game video clips (no search placeholders)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Lock
from time import monotonic
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import requests

_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball'
_cache = {}
_CACHE_MAX = 512  # highlights fan out ~50 summaries per league; leave headroom
_lock = Lock()


class ProviderUnavailable(RuntimeError):
    pass


def _fetch(league, path, **params):
    return _fetch_url(f'{_BASE}/{league}/{path}', **params)


def _fetch_url(url, ttl=900, **params):
    key = (url, tuple(sorted(params.items())))
    with _lock:
        cached = _cache.get(key)
        if cached and monotonic() - cached[0] < ttl:
            return cached[1]
    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError('Invalid ESPN response')
    except (requests.RequestException, ValueError) as exc:
        raise ProviderUnavailable('ESPN is temporarily unavailable') from exc
    with _lock:
        now = monotonic()
        _cache[key] = (now, payload)
        if len(_cache) > _CACHE_MAX:
            # Request params come from callers, so bound memory: drop the oldest entries first.
            for stale in sorted(_cache, key=lambda k: _cache[k][0])[:len(_cache) - _CACHE_MAX]:
                del _cache[stale]
    return payload


def _score(value):
    if isinstance(value, dict):
        value = value.get('value', value.get('displayValue'))
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def normalize_abbrev(league, abbrev):
    """Map ESPN's display codes to the codes used by app datasets and logos."""
    aliases = ({'NY': 'NYL', 'GS': 'GSV', 'GSW': 'GSV', 'LA': 'LAS', 'LV': 'LVA', 'WSH': 'WAS', 'POR': 'PDX'}
               if league == 'wnba' else
               {'PHO': 'PHX', 'NY': 'NYK', 'GS': 'GSW', 'NO': 'NOP', 'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS', 'NJ': 'BKN'})
    return aliases.get(abbrev, abbrev)


def _matchup(league, event):
    competitors = (event.get('competitions') or [{}])[0].get('competitors', [])
    home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
    away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
    if home and away:
        return f"{normalize_abbrev(league, away.get('team', {}).get('abbreviation', ''))} @ {normalize_abbrev(league, home.get('team', {}).get('abbreviation', ''))}"
    return event.get('shortName', event.get('name', ''))


def _team_schedule(team, league, season, seasons):
    abbrev = normalize_abbrev(league, team['abbreviation'])
    result = dict(abbrev=abbrev, name=team.get('displayName', abbrev),
                  nickname=team.get('name', team.get('shortDisplayName', '')),
                  city=team.get('location', ''), conference=None, division=None,
                  color=team.get('color', '64748b'),
                  logo=(team.get('logos') or [{}])[0].get('href'), season=season,
                  seasons=seasons, wins=None, losses=None, points_per_game=None,
                  points_allowed_per_game=None, history_through=None,
                  stats_available=False, games=[], upcoming=[])
    try:
        payload = _fetch(league, f"teams/{team['id']}/schedule", season=season, seasontype=2)
    except ProviderUnavailable:
        return result
    groups = payload.get('team', {}).get('groups', {})
    if league == 'wnba':
        result['conference'] = {'1': 'Eastern', '2': 'Western'}.get(str(groups.get('id')))
    else:
        divisions = {
            'Atlantic': {'BOS', 'BKN', 'NY', 'NYK', 'PHI', 'TOR'},
            'Central': {'CHI', 'CLE', 'DET', 'IND', 'MIL'},
            'Southeast': {'ATL', 'CHA', 'MIA', 'ORL', 'WSH', 'WAS'},
            'Northwest': {'DEN', 'MIN', 'OKC', 'POR', 'UTAH', 'UTA'},
            'Pacific': {'GS', 'GSW', 'LAC', 'LAL', 'PHX', 'SAC'},
            'Southwest': {'DAL', 'HOU', 'MEM', 'NO', 'NOP', 'SA', 'SAS'},
        }
        result['division'] = next((name for name, members in divisions.items() if abbrev in members), None)
        if result['division']:
            result['conference'] = 'Eastern' if result['division'] in ('Atlantic', 'Central', 'Southeast') else 'Western'
    for event in payload.get('events', []):
        # Never mix preseason / playoff scores into regular-season averages.
        kind = event.get('seasonType', {}).get('type', 2)
        if str(kind) != '2':
            continue
        competition = (event.get('competitions') or [{}])[0]
        # ESPN labels the Cup championship seasonType=2 even though it does
        # not count in league standings. Cup qualifying games remain standard
        # regular-season games and must still count.
        competition_type = competition.get('type', {})
        cup_final = competition_type.get('slug') in ('commissioners-cup', 'nba-cup-final', 'in-season-tournament-championship')
        notes = ' '.join(str(note.get('headline', '')) for note in competition.get('notes', [])).lower()
        if cup_final or ('cup' in notes and ('championship' in notes or 'final' in notes)):
            continue
        competitors = competition.get('competitors', [])
        own = next((c for c in competitors if str(c.get('id')) == str(team['id'])), None)
        other = next((c for c in competitors if str(c.get('id')) != str(team['id'])), None)
        if not own or not other:
            continue
        completed = competition.get('status', {}).get('type', {}).get('completed', False)
        pf, pa = _score(own.get('score')), _score(other.get('score'))
        game = dict(id=str(event['id']), date=event.get('date'),
                    opponent=normalize_abbrev(league, other.get('team', {}).get('abbreviation', '')),
                    is_home=own.get('homeAway') == 'home',
                    points_for=pf if completed else None, points_against=pa if completed else None,
                    result=('W' if pf > pa else 'L') if completed and pf is not None and pa is not None else None)
        if completed and game['result']:
            result['games'].append(game)
        elif not completed:
            result['upcoming'].append(game)
    result['games'].sort(key=lambda g: g['date'] or '', reverse=True)
    result['upcoming'].sort(key=lambda g: g['date'] or '')
    games = result['games']
    if games:
        result.update(wins=sum(g['result'] == 'W' for g in games), losses=sum(g['result'] == 'L' for g in games),
                      points_per_game=round(sum(g['points_for'] for g in games) / len(games), 1),
                      points_allowed_per_game=round(sum(g['points_against'] for g in games) / len(games), 1),
                      history_through=games[0]['date'], stats_available=True)
    return result


def get_teams(league, season=None, abbrev=None):
    try:
        payload = _fetch(league, 'teams', limit=100)
        metadata = payload['sports'][0]['leagues'][0]
        current = int(metadata.get('season', {}).get('year') or datetime.now(ZoneInfo('America/New_York')).year)
        season = season or current
        seasons = list(range(current, current - 3, -1))
        teams = [entry['team'] for entry in metadata.get('teams', []) if not entry['team'].get('isAllStar')]
        if abbrev:
            teams = [t for t in teams if normalize_abbrev(league, t['abbreviation'].upper()) == normalize_abbrev(league, abbrev.upper())]
        with ThreadPoolExecutor(max_workers=8) as pool:
            profiles = list(pool.map(lambda t: _team_schedule(t, league, season, seasons), teams))
        return dict(league=league, season=season, seasons=seasons, teams=profiles,
                    message=None if any(t['stats_available'] for t in profiles) else 'No completed regular-season scores available for this season.')
    except (ProviderUnavailable, KeyError, IndexError, TypeError, ValueError):
        return dict(league=league, season=season, seasons=[], teams=[], message='Team data is temporarily unavailable from ESPN.')


def _safe_url(value, video=False):
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    host = parsed.hostname or ''
    allowed = host == 'espn.com' or host.endswith('.espn.com') or host.endswith('.espncdn.com') or host == 'espnmedia-cdn.akamaized.net'
    if parsed.scheme == 'https' and allowed and (not video or parsed.path.endswith('.mp4')):
        return value
    return None


def _clips(payload):
    clips = []
    for clip in payload.get('videos', []):
        links = clip.get('links') or {}
        source = links.get('source') or {}
        media = _safe_url(source.get('href'), video=True) or _safe_url((source.get('HD') or {}).get('href'), video=True)
        external = _safe_url((links.get('web') or {}).get('href'))
        clip_id = str(clip.get('id', ''))
        if not external and clip_id.isdigit():
            external = f'https://www.espn.com/video/clip/_/id/{clip_id}'
        if not media and not external:
            continue
        clips.append(dict(id=clip_id, title=clip.get('headline', 'Game highlight'),
                          description=clip.get('description', ''), thumbnail=_safe_url(clip.get('thumbnail')),
                          url=media or external, video_url=media, external_url=external,
                          duration=clip.get('duration'), source='ESPN'))
    return clips


def get_highlights(league):
    now = datetime.now(ZoneInfo('America/New_York'))
    checked = now.isoformat()
    events, failures = {}, 0
    def board(offset):
        day = (now + timedelta(days=offset)).strftime('%Y%m%d')
        try:
            return _fetch(league, 'scoreboard', dates=day).get('events', [])
        except ProviderUnavailable:
            return None
    with ThreadPoolExecutor(max_workers=5) as pool:
        boards = list(pool.map(board, range(-3, 2)))
    for board_events in boards:
        if board_events is None:
            failures += 1
        for event in board_events or []:
            if event.get('status', {}).get('type', {}).get('state') in ('post', 'in'):
                events[str(event['id'])] = event
    def game(event):
        try:
            clips = _clips(_fetch(league, 'summary', event=event['id']))
        except ProviderUnavailable:
            clips = []
        return dict(game_id=str(event['id']), matchup=_matchup(league, event),
                    date=event.get('date'), status=event.get('status', {}).get('type', {}).get('description', ''), clips=clips)
    with ThreadPoolExecutor(max_workers=8) as pool:
        games = list(pool.map(game, events.values()))
    games.sort(key=lambda g: g['date'] or '', reverse=True)
    message = None if any(g['clips'] for g in games) else ('ESPN highlights are temporarily unavailable.' if failures == 5 else 'ESPN has not published clips for recent games yet.')
    return dict(league=league, checked_at=checked, games=games, message=message)
