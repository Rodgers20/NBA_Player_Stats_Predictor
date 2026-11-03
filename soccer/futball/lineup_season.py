"""Season totals for the last XI, from every covered appearance (including subs)."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor


def aggregate(summaries: list[dict], team: str, expected: int) -> tuple[dict, int]:
    from .match_lineups import canonical
    totals: dict = {}
    covered = 0
    seen = set()
    for body in summaries:
        event = (body.get('header') or {}).get('id')
        if not event or event in seen:
            continue
        seen.add(event)
        roster = next((r for r in body.get('rosters', []) if canonical(r.get('team', {}).get('displayName', '')) == canonical(team)), None)
        if not roster or sum(r.get('starter') is True for r in roster.get('roster', [])) != 11:
            continue
        covered += 1
        for row in roster['roster']:
            stats = {s['name']: s.get('value') for s in row.get('stats', [])}
            if not (row.get('starter') is True or row.get('subbedIn') is True or (stats.get('appearances') or 0) > 0):
                continue
            pid = str((row.get('athlete') or {}).get('id') or '')
            if not pid:
                continue
            t = totals.setdefault(pid, {'goals': 0, 'assists': 0, 'played': 0, 'ratings': []})
            t['played'] += 1
            for field, key in [('goals', 'totalGoals'), ('assists', 'goalAssists')]:
                value = stats.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    t[field] = None
                elif t[field] is not None:
                    t[field] += int(value)
            # A missing match rating is excluded, never replaced with zero or a
            # goals-based estimate. ESPN currently supplies no rating field.
            rating = stats.get('rating')
            if isinstance(rating, (int, float)) and math.isfinite(rating) and 0 < rating <= 10:
                t['ratings'].append(rating)
    for t in totals.values():
        ratings = t.pop('ratings')
        t['rated_matches'] = len(ratings)
        t['rating'] = round(sum(ratings) / len(ratings), 2) if ratings else None
        if covered != expected:
            t.update(goals=None, assists=None, rating=None)
    return totals, covered


def enrich(people: list[dict], team: str, code: str) -> dict:
    from . import config
    from .match_lineups import fixtures, finished, canonical, match_summary
    rows = fixtures(code)
    rows = [r for r in rows if finished(r) and canonical(team) in [canonical(r[s]['name']) for s in ('homeTeam', 'awayTeam')]]
    year = max((str((r.get('season') or {}).get('startDate') or '')[:4] for r in rows), default='')
    rows = [r for r in rows if str((r.get('season') or {}).get('startDate') or '')[:4] == year] if year else []
    # Deduplicate fixture records, not player names.
    rows = list({r['id']: r for r in rows}.values())
    with ThreadPoolExecutor(max_workers=3) as pool:
        summaries = list(pool.map(lambda r: match_summary(r, code), rows))
    totals, covered = aggregate(summaries, team, len(rows))
    for player in people:
        stats = totals.get(player['provider_id'], {})
        for key in ('goals', 'assists', 'played', 'rating'):
            player[key] = stats.get(key)
        player['rated_matches'] = stats.get('rated_matches', 0)
        player['stats_scope'] = 'season'
    season = (rows[0].get('season') or {}) if rows else {}
    start, end = season.get('startDate', '')[:4], season.get('endDate', '')[:4]
    from .season_ratings import enrich as enrich_ratings
    rating_info = enrich_ratings(people, team, code, f'{start}/{end}' if start and end else None)
    return {**rating_info, 'stats_competition': config.LEAGUES.get(code, {}).get('name', code),
            'stats_season': f'{start}/{end}' if start and end else None,
            'stats_matches_covered': covered, 'stats_matches_expected': len(rows),
            'stats_note': f'Season totals · {covered}/{len(rows)} team matches covered. Includes substitute appearances. Ratings average published match ratings only; — means unavailable.'}
