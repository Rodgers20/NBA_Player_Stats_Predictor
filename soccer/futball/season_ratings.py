"""Published FotMob season averages from public team pages; no invented ratings."""
from __future__ import annotations
import hashlib
import json
import math
import re
import threading
import unicodedata

import httpx
from . import config
from .sources import _cache

RAW = config.RAW / 'fotmob-ratings'
LEAGUES = {'SP1': (87, 'laliga'), 'E0': (47, 'premier-league')}
_lock = threading.Lock()


def key(name: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', name).casefold() if c.isalnum() and not unicodedata.combining(c))


def page(path: str) -> dict:
    if not re.fullmatch(r'/(teams|leagues)/\d+/overview/[a-z0-9-]+', path):
        return {}
    dest = RAW / (hashlib.sha256(path.encode()).hexdigest()[:20] + '.json')
    with _lock:
        cached = _cache.read_json(dest)
        if cached is not None and _cache.is_fresh(dest, 21600 if cached else 900):
            return cached
        try:
            response = httpx.get('https://www.fotmob.com' + path, timeout=25, follow_redirects=True)
            response.raise_for_status()
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', response.text, re.S)
            data = json.loads(match[1])['props']['pageProps']['fallback'] if match else {}
            _cache.write_json(dest, data)
            return data
        except (httpx.HTTPError, ValueError, KeyError, OSError):
            # Expired ratings are not silently presented as current averages.
            return {}


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def apply_ratings(data: dict, people: list[dict], team: str, code: str, season: str) -> int:
    from .match_lineups import canonical
    details = data.get('details') or {}
    overview = data.get('overview') or {}
    if (code not in LEAGUES or details.get('primaryLeagueId') != LEAGUES[code][0]
            or overview.get('season') != season or canonical(details.get('name', '')) != canonical(team)):
        return 0
    members = [p for group in (data.get('squad') or {}).get('squad', []) if group.get('title') != 'coach' for p in group.get('members', [])]
    count = 0
    for player in people:
        matches = [p for p in members if key(p.get('name', '')) == key(player['name']) and str(p.get('shirtNumber')) == str(player.get('shirt'))]
        if len(matches) != 1:
            continue
        row = matches[0]
        rating = row.get('rating')
        if not isinstance(rating, (int, float)) or isinstance(rating, bool) or not math.isfinite(rating) or not 0 < rating <= 10:
            continue
        player.update(rating=round(rating, 2), rating_source='FotMob', rated_matches=None,
                      rating_source_url=f'https://www.fotmob.com/players/{row["id"]}')
        count += 1
    return count


def enrich(people: list[dict], team: str, code: str, season: str | None) -> dict:
    from .match_lineups import canonical
    if code not in LEAGUES or not season:
        return {}
    league_id, slug = LEAGUES[code]
    # Public team overview pages embed the full domestic league table; the
    # league landing page itself may defer that table to a browser request.
    anchor = {'SP1': '/teams/8634/overview/barcelona', 'E0': '/teams/9825/overview/arsenal'}[code]
    directory = page(anchor)
    paths = {r['pageUrl'] for r in objects(directory) if isinstance(r.get('pageUrl'), str)
             and re.fullmatch(r'/teams/\d+/overview/[a-z0-9-]+', r['pageUrl'])
             and canonical(r.get('name', '')) == canonical(team)}
    if len(paths) != 1:
        return {}
    path = paths.pop()
    team_id = path.split('/')[2]
    data = page(path).get(f'team-{team_id}', {})
    count = apply_ratings(data, people, team, code, season)
    return {'ratings_source': 'FotMob', 'ratings_source_url': 'https://www.fotmob.com' + path,
            'ratings_note': f'FotMob season averages · {count}/{len(people)} players matched. Updated from the published team page; unrated players show —.'} if count else {}
