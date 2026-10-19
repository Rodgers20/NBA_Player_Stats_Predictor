from utils import basketball_boxscore as bs

LABELS = ['MIN', 'PTS', 'REB', 'AST']


def athlete(pid, name, stats, starter=False, dnp=False, reason=None):
    return dict(athlete=dict(id=pid, displayName=name, shortName=name[0] + '. ' + name.split()[-1],
                             position=dict(abbreviation='G'), jersey='1'),
                stats=stats, starter=starter, didNotPlay=dnp, reason=reason)


def summary(state='post'):
    def side(tid, abbrev, home, score):
        return dict(homeAway=home, score=str(score), winner=score > 100, team=dict(id=tid, abbreviation=abbrev, displayName=abbrev),
                    linescores=[dict(displayValue='30'), dict(displayValue='25')],
                    record=[dict(type='total', displayValue='1-0')])
    return dict(
        header=dict(id='401', competitions=[dict(id='401', status=dict(type=dict(state=state, detail='Final'), period=4, displayClock='0:00'),
                                                  competitors=[side('1', 'TOR', 'away', 98), side('2', 'CLE', 'home', 126)])]),
        boxscore=dict(
            players=[dict(team=dict(id='1'), statistics=[dict(labels=LABELS, totals=['', '98', '40', '20'], athletes=[
                athlete('9', 'Brandon Ingram', ['36', '17', '2', '4'], starter=True),
                athlete('8', 'Bench Guy', ['20', '21', '9', '1']),
                athlete('7', 'Out Player', [], dnp=True, reason="COACH'S DECISION")])]),
                dict(team=dict(id='2'), statistics=[dict(labels=LABELS, totals=['', '126', '45', '30'], athletes=[
                    athlete('5', 'Donovan Mitchell', ['30', '40', '5', '8'], starter=True)])])],
            teams=[dict(team=dict(id='1'), statistics=[dict(label='FG', displayValue='37-71'), dict(label='Bogus', displayValue='1')]),
                   dict(team=dict(id='2'), statistics=[])]))


def test_parses_players_totals_and_linescore():
    box = bs.parse_boxscore('nba', summary())
    away, home = box['teams']
    assert (away['abbrev'], away['score'], home['score'], home['winner']) == ('TOR', 98, 126, True)
    assert away['linescore'] == [30, 25] and away['record'] == '1-0'
    assert away['totals']['PTS'] == '98' and away['team_stats'] == [dict(label='FG', value='37-71')]
    assert away['players'][2]['dnp'] and away['players'][2]['reason'] == "COACH'S DECISION"
    assert box['available']


def test_players_carry_headshots():
    away = bs.parse_boxscore('wnba', summary())['teams'][0]
    assert away['players'][0]['image'] == 'https://a.espncdn.com/i/headshots/wnba/players/full/9.png'
    assert away['leaders'][0]['image'].endswith('/8.png')


def test_leaders_skip_dnp_and_pick_max():
    away = bs.parse_boxscore('nba', summary())['teams'][0]
    assert {l['label']: l['name'] for l in away['leaders']} == {'PTS': 'B. Guy', 'REB': 'B. Guy', 'AST': 'B. Ingram'}


def test_pregame_has_no_box_score(monkeypatch):
    monkeypatch.setattr(bs, '_fetch_url', lambda *a, **k: summary('pre') | dict(boxscore={}))
    result = bs.get_boxscore('nba', '401869187')
    assert not result['available'] and result['teams'][0]['score'] is None and 'tips off' in result['message']


def test_provider_failure_is_reported(monkeypatch):
    def boom(*a, **k):
        raise bs.ProviderUnavailable('x')
    monkeypatch.setattr(bs, '_fetch_url', boom)
    assert 'temporarily unavailable' in bs.get_boxscore('nba', '401869187')['message']
