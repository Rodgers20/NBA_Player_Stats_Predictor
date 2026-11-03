"""Cached provider biographies and recorded career teams for player profiles.

Market values, preferred foot and contract dates are deliberately not inferred
from transfer fees, shot events, team seasons, or the player's age.
"""
from pathlib import Path
import threading
import time
from urllib.parse import urlparse
from . import config, headshots, players
from .sources import api_football as af, _cache

_LOCK = threading.Lock()
TTL = 7 * 86400
RETRY_TTL = 1800


def provider_id(name: str) -> int | None:
    url = headshots.portrait_url(name)
    if not url:
        return None
    stem = Path(urlparse(url).path).stem
    return int(stem) if stem.isdigit() else None


def normalize(player_id: int, profile_response: dict, teams_response: dict) -> dict:
    profiles = profile_response.get('response') or []
    player = next((r.get('player') or {} for r in profiles
                   if (r.get('player') or {}).get('id') == player_id), {})
    career = []
    for row in teams_response.get('response') or []:
        team = row.get('team') or {}
        if not team.get('id') or not team.get('name'):
            continue
        seasons = sorted({s for s in (row.get('seasons') or []) if isinstance(s,int) and not isinstance(s,bool)},reverse=True)
        career.append(dict(team_id=team['id'],team=team['name'],crest=team.get('logo'),seasons=seasons))
    career.sort(key=lambda r: (-max(r['seasons'],default=0),r['team']))
    return dict(provider_id=player_id,source='API-Football',
                height=player.get('height'),weight=player.get('weight'),
                date_of_birth=(player.get('birth') or {}).get('date'),
                nationality=player.get('nationality'),
                preferred_foot=None,contract_end=None,market_value=None,
                value_history=[],career=career)


def details(public_id: str) -> dict | None:
    key = players._by_public_id().get(str(public_id))
    if key is None:
        return None
    name = players._identities()[key]['name']
    pid = provider_id(name)
    empty = dict(provider_id=pid,source='API-Football',height=None,weight=None,
                 date_of_birth=None,nationality=None,preferred_foot=None,
                 contract_end=None,market_value=None,value_history=[],career=[],
                 fetched_at=None,status='unavailable')
    if pid is None:
        return {**empty,'status':'unmatched'}
    path = config.RAW/'api-football'/'player-details'/f'{pid}.json'
    with _LOCK:
        cached = _cache.read_json(path)
        if cached and _cache.is_fresh(path,TTL if cached.get('status')=='available' else RETRY_TTL):
            return cached
        if not af.available():
            return cached or {**empty,'status':'not_configured'}
        remaining = af.requests_remaining()
        if remaining is None or remaining < 5:
            return {**(cached or empty),'status':'quota_unavailable'}
        responses={}; failed=False
        for endpoint in ('profiles','teams'):
            try:
                responses[endpoint]=af._get('/players/'+endpoint,{'player':pid})
            except Exception:
                responses[endpoint]={};failed=True
        result=normalize(pid,responses['profiles'],responses['teams'])
        if failed and cached:
            return {**cached,'status':'stale'}
        result.update(fetched_at=time.time(),status='partial' if failed else 'available')
        _cache.write_json(path,result)
        return result
