"""ESPN standings, power rankings and the playoff picture for NBA / WNBA."""
from datetime import datetime
from zoneinfo import ZoneInfo
from utils.basketball_content import ProviderUnavailable, normalize_abbrev, _fetch_url

_URL = 'https://site.web.api.espn.com/apis/v2/sports/basketball/{league}/standings'
# direct playoff seeds, then play-in seeds (NBA only)
_FORMAT = {'nba': (6, 4), 'wnba': (8, 0)}
_FORM_WEIGHT = 0.3  # power points per win above .500 in the last ten games


def _num(value, default=None):
    try:
        return float(str(value).replace('+', ''))
    except (TypeError, ValueError):
        return default


def _record(text):
    try:
        wins, losses = str(text).split('-')[:2]
        return int(wins), int(losses)
    except ValueError:
        return 0, 0


def _row(league, conference, entry):
    stats = {s.get('name'): s.get('displayValue') for s in entry.get('stats', [])}
    team = entry.get('team', {})
    wins, losses = int(_num(stats.get('wins'), 0)), int(_num(stats.get('losses'), 0))
    last10 = _record(stats.get('Last Ten Games'))
    games = wins + losses
    return dict(
        abbrev=normalize_abbrev(league, team.get('abbreviation', '')), name=team.get('displayName', ''),
        nickname=team.get('name', ''), logo=(team.get('logos') or [{}])[0].get('href'),
        conference=conference, wins=wins, losses=losses,
        win_pct=round(wins / games, 3) if games else None,
        games_behind=stats.get('gamesBehind') or '-',
        points_for=_num(stats.get('avgPointsFor')), points_against=_num(stats.get('avgPointsAgainst')),
        point_diff=_num(stats.get('differential')),
        home=stats.get('Home'), road=stats.get('Road'), conf=stats.get('vs. Conf.'), div=stats.get('vs. Div.'),
        last10=stats.get('Last Ten Games'), last10_wins=last10[0] if sum(last10) else None,
        streak=stats.get('streak'), clincher=stats.get('clincher') or None,
        seed=int(_num(stats.get('playoffSeed'), 0)) or None, games=games)


def parse_standings(league, payload):
    """Conference tables ordered by seed, then win percentage."""
    conferences = []
    for child in payload.get('children', []):
        name = child.get('name', '').replace(' Conference', '')
        rows = [_row(league, name, e) for e in child.get('standings', {}).get('entries', [])]
        rows.sort(key=lambda r: (r['seed'] or 99, -(r['win_pct'] or 0)))
        for index, row in enumerate(rows, 1):
            row['seed'] = row['seed'] or index
        conferences.append(dict(name=name, teams=rows))
    return conferences


def power_rankings(conferences):
    """Net points per game plus a recent-form bump; movement compares with the season-only rank."""
    rows = [dict(r) for c in conferences for r in c['teams'] if r['games'] and r['point_diff'] is not None]
    for row in rows:
        form = (row['last10_wins'] - 5) * _FORM_WEIGHT if row['last10_wins'] is not None else 0
        row['power'] = round(row['point_diff'] + form, 2)
    by_season = {r['abbrev']: i for i, r in enumerate(sorted(rows, key=lambda r: -r['point_diff']), 1)}
    ranked = sorted(rows, key=lambda r: -r['power'])
    return [dict(rank=i, abbrev=r['abbrev'], name=r['name'], nickname=r['nickname'], logo=r['logo'],
                 wins=r['wins'], losses=r['losses'], power=r['power'], point_diff=r['point_diff'],
                 last10=r['last10'], streak=r['streak'], movement=by_season[r['abbrev']] - i)
            for i, r in enumerate(ranked, 1)]


def playoff_picture(league, conferences):
    direct, play_in = _FORMAT[league]
    picture = []
    for conference in conferences:
        teams = conference['teams']
        cutoff = teams[direct - 1] if len(teams) >= direct else None
        entries = []
        for team in teams:
            zone = 'direct' if team['seed'] <= direct else 'play-in' if team['seed'] <= direct + play_in else 'out'
            gap = None
            if cutoff and zone != 'direct':
                gap = ((cutoff['wins'] - team['wins']) + (team['losses'] - cutoff['losses'])) / 2
            entries.append(dict(abbrev=team['abbrev'], nickname=team['nickname'], logo=team['logo'], seed=team['seed'],
                                wins=team['wins'], losses=team['losses'], clincher=team['clincher'], zone=zone,
                                games_back_of_direct=gap))
        picture.append(dict(name=conference['name'], teams=entries))
    return dict(direct=direct, play_in=play_in, conferences=picture)


def _current_season(league, now):
    # The NBA season is labelled by the year it ends; it starts in October.
    return now.year + (1 if league == 'nba' and now.month >= 10 else 0)


def get_standings(league, season=None):
    """Falls back to the previous season until the current one has results."""
    current = _current_season(league, datetime.now(ZoneInfo('America/New_York')))
    empty = dict(league=league, season=season or current, conferences=[], power_rankings=[],
                 playoff_picture=None, fallback=False)
    try:
        for candidate in ([season] if season else [current, current - 1]):
            conferences = parse_standings(league, _fetch_url(_URL.format(league=league), ttl=300, season=candidate, seasontype=2))  # 2 = regular season; ESPN's default includes preseason
            if any(t['games'] for c in conferences for t in c['teams']):
                return dict(empty, season=candidate, conferences=conferences,
                            power_rankings=power_rankings(conferences),
                            playoff_picture=playoff_picture(league, conferences),
                            fallback=candidate != current, message=None)
        return dict(empty, message='No completed games yet this season.')
    except (ProviderUnavailable, KeyError, TypeError, ValueError):
        return dict(empty, message='Standings are temporarily unavailable from ESPN.')
