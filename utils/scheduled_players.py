"""Date-scoped player suggestions from free ESPN schedules and team rosters."""
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import requests

_CACHE = {}
_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball'
_OUT = {'out', 'injured reserve', 'suspended', 'inactive'}


def _fetch(league, path, params=None):
    key = (league, path, tuple(sorted((params or {}).items())))
    cached = _CACHE.get(key)
    if cached and time.monotonic() - cached[0] < 300:
        return cached[1]
    response = requests.get(f'{_BASE}/{league}/{path}', params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    # Bound cache growth when browsing many historical dates.
    if len(_CACHE) >= 128:
        _CACHE.clear()
    _CACHE[key] = (time.monotonic(), data)
    return data


def _abbreviation(team, league):
    raw = str(team.get('abbreviation') or '').upper()
    if league == 'wnba':
        return {'NY': 'NYL', 'LA': 'LAS', 'LV': 'LVA', 'GS': 'GSV',
                'GSW': 'GSV', 'WSH': 'WAS', 'PHO': 'PHX'}.get(raw, raw)
    return {'NY': 'NYK', 'GS': 'GSW', 'SA': 'SAS', 'NO': 'NOP',
            'UTAH': 'UTA', 'WSH': 'WAS', 'PHO': 'PHX'}.get(raw, raw)


def scheduled_players(league, game_date):
    """Never substitute tomorrow's slate or the full historical player list."""
    result = dict(league=league, game_date=game_date, players=[])
    try:
        board = _fetch(league, 'scoreboard', {'dates': game_date.replace('-', '')})
        teams = {}
        for event in board.get('events', []):
            # ESPN event times are UTC; a late tip can fall on the next UTC day.
            tip = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
            if tip.astimezone(ZoneInfo('America/New_York')).date().isoformat() != game_date:
                continue
            for competition in event.get('competitions', []):
                status = str((competition.get('status') or event.get('status') or {}).get('type', {}).get('name', '')).lower()
                if 'postpon' in status or 'cancel' in status:
                    continue
                competitors = competition.get('competitors', [])
                if len(competitors) != 2:
                    continue
                for index, competitor in enumerate(competitors):
                    team = competitor['team']
                    teams[str(team['id'])] = (
                        _abbreviation(team, league),
                        _abbreviation(competitors[1-index]['team'], league))
        if not teams:
            return dict(result, message='No games scheduled for this date.')
        players = {}
        for team_id, (team, opponent) in teams.items():
            roster = _fetch(league, f'teams/{team_id}/roster')
            for athlete in roster.get('athletes', []):
                status = athlete.get('status') or {}
                status_name = status.get('type') or status.get('name') or '' if isinstance(status, dict) else str(status)
                injuries = athlete.get('injuries') or []
                if str(status_name).lower() in _OUT or any(str(item.get('status', '')).lower() in _OUT for item in injuries):
                    continue
                name = athlete.get('displayName') or athlete.get('fullName')
                if name:
                    players[name] = dict(name=name, team=team, opponent=opponent)
        return dict(result, players=sorted(players.values(), key=lambda player: player['name']),
                    message='Current rosters for scheduled teams; reported out players excluded. Lineups are not confirmed.')
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return dict(result, message='Schedule or roster data is unavailable for this date. Try again later.')
