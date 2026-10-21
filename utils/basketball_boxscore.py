"""Game box scores (player lines, team totals, line score) from ESPN's summary feed."""
from utils.basketball_content import _BASE, ProviderUnavailable, normalize_abbrev, _fetch_url, _safe_url

_TEAM_STATS = ('FG', 'Field Goal %', '3PT', 'Three Point %', 'FT', 'Free Throw %', 'Rebounds',
               'Offensive Rebounds', 'Assists', 'Steals', 'Blocks', 'Total Turnovers',
               'Points in Paint', 'Fast Break Points', 'Largest Lead')
_LEADERS = (('PTS', 'Points'), ('REB', 'Rebounds'), ('AST', 'Assists'))


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _image(league, athlete):
    """ESPN headshot; falls back to the CDN path ESPN uses for every player id."""
    url = _safe_url((athlete.get('headshot') or {}).get('href'))
    pid = str(athlete.get('id', ''))
    return url or (f'https://a.espncdn.com/i/headshots/{league}/players/full/{pid}.png' if pid.isdigit() else None)


def _player(league, athlete_entry, labels):
    athlete = athlete_entry.get('athlete', {})
    stats = dict(zip(labels, athlete_entry.get('stats') or []))
    return dict(id=str(athlete.get('id', '')), name=athlete.get('displayName', ''), image=_image(league, athlete),
                short=athlete.get('shortName', athlete.get('displayName', '')),
                position=(athlete.get('position') or {}).get('abbreviation', ''), jersey=athlete.get('jersey', ''),
                starter=bool(athlete_entry.get('starter')), dnp=bool(athlete_entry.get('didNotPlay')),
                reason=athlete_entry.get('reason') or None, stats=stats)


def _leaders(players):
    leaders = []
    for label, title in _LEADERS:
        played = [p for p in players if not p['dnp'] and label in p['stats']]
        if not played:
            continue
        best = max(played, key=lambda p: _int(p['stats'][label]))
        leaders.append(dict(label=label, title=title, name=best['short'], value=_int(best['stats'][label]), id=best['id'], image=best['image']))
    return leaders


def parse_boxscore(league, summary):
    competition = (summary.get('header', {}).get('competitions') or [{}])[0]
    status = competition.get('status', {})
    state = status.get('type', {}).get('state', 'pre')
    competitors = {c.get('homeAway'): c for c in competition.get('competitors', [])}
    box = summary.get('boxscore', {})
    groups = {g.get('team', {}).get('id'): g for g in box.get('players', [])}
    totals = {t.get('team', {}).get('id'): t for t in box.get('teams', [])}
    teams = []
    for side in ('away', 'home'):
        competitor = competitors.get(side)
        if not competitor:
            continue
        team = competitor.get('team', {})
        group = (groups.get(team.get('id'), {}).get('statistics') or [{}])[0]
        labels = group.get('labels') or []
        players = [_player(league, a, labels) for a in group.get('athletes', [])]
        stats = {s.get('label'): s.get('displayValue') for s in totals.get(team.get('id'), {}).get('statistics', [])}
        teams.append(dict(
            side=side, abbrev=normalize_abbrev(league, team.get('abbreviation', '')), name=team.get('displayName', ''),
            logo=team.get('logo'), color=team.get('color'), score=_int(competitor.get('score')) if state != 'pre' else None,
            winner=bool(competitor.get('winner')),
            record=next((r.get('displayValue') for r in competitor.get('record', []) if r.get('type') == 'total'), None),
            linescore=[_int(p.get('displayValue')) for p in competitor.get('linescores', [])],
            columns=labels, players=players, totals=dict(zip(labels, group.get('totals') or [])),
            team_stats=[dict(label=k, value=stats[k]) for k in _TEAM_STATS if k in stats],
            leaders=_leaders(players)))
    return dict(game_id=str(competition.get('id') or summary.get('header', {}).get('id', '')), state=state,
                status=status.get('type', {}).get('detail', ''), period=status.get('period'),
                clock=status.get('displayClock'), teams=teams,
                available=state != 'pre' and any(t['players'] for t in teams))


def get_boxscore(league, game_id):
    try:
        data = parse_boxscore(league, _fetch_url(f'{_BASE}/{league}/summary', ttl=20, event=game_id))
    except (ProviderUnavailable, KeyError, TypeError, ValueError):
        return dict(game_id=str(game_id), state='unknown', status='', teams=[], available=False,
                    message='Box score is temporarily unavailable from ESPN.')
    if not data['available']:
        data['message'] = 'The box score appears once the game tips off.'
    return data
