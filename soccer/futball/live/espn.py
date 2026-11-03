"""Shared, demand-driven ESPN score polling. Never infer live status from kickoff."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import threading
import time
import httpx
from .. import config
from ..sources import _cache

LEAGUES = config.ESPN_SLUGS
RAW = config.RAW / 'espn-live'
_locks = {code: threading.Lock() for code in LEAGUES}
_memory = {}


def parse(body: dict, code: str, now: datetime) -> list[dict]:
    from ..league import _to_lake_name
    out = []
    seen = set()
    for event in body.get('events', []):
        eid = str(event.get('id') or '')
        if not eid or eid in seen:
            continue
        seen.add(eid)
        comp = (event.get('competitions') or [{}])[0]
        status = comp.get('status') or event.get('status') or {}
        kind = status.get('type') or {}
        name = kind.get('name', '')
        state = kind.get('state')
        live = state == 'in' and name not in ('STATUS_POSTPONED','STATUS_CANCELED','STATUS_SUSPENDED','STATUS_ABANDONED','STATUS_DELAYED')
        kickoff = event.get('date') or ''
        if kickoff[:10] != now.date().isoformat() and not live:
            continue
        sides = {r.get('homeAway'):r for r in comp.get('competitors', [])}
        if not all(s in sides for s in ('home','away')):
            continue
        def team(side):
            raw = sides[side].get('team', {}).get('displayName', '')
            return _to_lake_name(raw, code) or raw
        def score(side):
            raw = sides[side].get('score')
            if state not in ('in','post') or raw is None:
                return None
            try:
                n = int(raw)
                return n if n >= 0 else None
            except (ValueError,TypeError):
                return None
        home, away = team('home'), team('away')
        out.append({'provider_id':eid,'competition_code':code,'date':kickoff[:10],
            'time':kickoff[11:16], 'kickoff_utc':kickoff,'home_team':home,'away_team':away,
            'fixture_key':f'{code}|espn|{eid}', 'minutes_elapsed':None,
            'home_goals':score('home'),'away_goals':score('away'),
            'provider_status':name, 'status_label':kind.get('shortDetail') or status.get('displayClock') or kind.get('description') or 'Unknown',
            'is_live':live,'completed':bool(kind.get('completed')), 'source':'ESPN',
            'source_url':f'https://www.espn.com/soccer/match/_/gameId/{eid}'})
    return sorted(out,key=lambda r:r['kickoff_utc'])


def _scoreboard(code: str, now: datetime) -> dict:
    """Yesterday + today. ESPN rejects any multi-day `dates` range with a 400, so ask per day."""
    events: list = []
    for day in (now - timedelta(days=1), now):
        response = httpx.get(f'https://site.api.espn.com/apis/site/v2/sports/soccer/{LEAGUES[code]}/scoreboard',
                             params={'dates': f'{day:%Y%m%d}', 'limit': 100}, timeout=12)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get('events'), list):
            raise ValueError('Invalid scoreboard')
        events.extend(body['events'])
    return {'events': events}


def board(code: str) -> dict:
    now = datetime.now(timezone.utc)
    day = now.date().isoformat()
    with _locks[code]:
        cached = _memory.get(code)
        if cached and cached['day'] == day and time.monotonic() < cached['retry_at']:
            return cached['result']
        path = RAW / f'{code}-{day}.json'
        disk = _cache.read_json(path) or {}
        fetched = disk.get('fetched_at')
        try:
            recent = (now-datetime.fromisoformat(fetched)).total_seconds() < disk.get('ttl',45) if fetched else False
        except (TypeError,ValueError):
            recent = False
        try:
            if recent:
                body = disk['body']
            else:
                body = _scoreboard(code, now)
                fetched = now.isoformat()
            rows = parse(body, code, now)
            # Discover delayed starts quickly near kickoff; idle leagues poll slowly.
            soon = any(r['kickoff_utc'] and abs((datetime.fromisoformat(r['kickoff_utc'].replace('Z','+00:00'))-now).total_seconds()) < 900 for r in rows)
            ttl = 45 if any(r['is_live'] for r in rows) or soon else 300
            result = {'competition_code':code,'matches':rows,'updated_at':fetched,'stale':False,'error':None}
            if not recent:
                try:
                    _cache.write_json(path,{'body':body,'fetched_at':fetched,'ttl':ttl})
                except OSError:
                    pass  # Memory sharing still works if the external disk is absent.
        except (httpx.HTTPError,ValueError,KeyError,TypeError):
            rows = parse(disk.get('body') or {},code,now)
            result = {'competition_code':code,'matches':rows,'updated_at':fetched,'stale':True,'error':'ESPN refresh unavailable'}
            ttl = 45
        # Do not extend a disk cache's original freshness on a process restart.
        remaining = max(1,ttl-(now-datetime.fromisoformat(fetched)).total_seconds()) if recent and fetched else ttl
        _memory[code] = {'day':day,'retry_at':time.monotonic()+remaining,'result':result}
        return result


def snapshot(code: str | None = None) -> dict:
    codes = [code] if code in LEAGUES else list(LEAGUES) if code is None else []
    with ThreadPoolExecutor(max_workers=3) as pool:
        boards = list(pool.map(board,codes))
    rows = [{**r,'updated_at':b['updated_at'],'stale':b['stale']} for b in boards for r in b['matches']]
    return {'source':'ESPN','matches':rows,'competitions':boards,
            'stale':not boards or any(b['stale'] for b in boards),'poll_seconds':45}


def counts() -> dict:
    data = snapshot()
    rows = [r for r in data['matches'] if r['is_live'] and not r['stale']]
    per = {c:sum(r['competition_code']==c for r in rows) for c in LEAGUES}
    return {'total':len(rows),'matches':rows,'by_competition':per,'stale':data['stale'],'source':'ESPN'}
