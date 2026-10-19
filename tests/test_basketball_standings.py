from utils import basketball_standings as st


def entry(abbrev, wins, losses, diff, seed, last10='5-5'):
    stats = dict(wins=wins, losses=losses, differential=diff, playoffSeed=seed, streak='W1')
    stats['Last Ten Games'] = last10
    return dict(team=dict(abbreviation=abbrev, displayName=abbrev, name=abbrev, logos=[{'href': 'x'}]),
                stats=[dict(name=k, displayValue=str(v)) for k, v in stats.items()])


def payload():
    east = [entry('AAA', 50, 32, '+3.0', 1, '5-5'), entry('BBB', 49, 33, '+6.0', 2, '9-1'), entry('CCC', 20, 62, '-9.0', 3)]
    return dict(children=[dict(name='Eastern Conference', standings=dict(entries=east))])


def test_parse_orders_by_seed_and_reads_form():
    (east,) = st.parse_standings('nba', payload())
    assert east['name'] == 'Eastern'
    assert [t['abbrev'] for t in east['teams']] == ['AAA', 'BBB', 'CCC']
    assert east['teams'][1]['last10_wins'] == 9 and east['teams'][1]['point_diff'] == 6.0


def test_power_rankings_blend_form_and_report_movement():
    ranks = st.power_rankings(st.parse_standings('nba', payload()))
    assert [r['abbrev'] for r in ranks] == ['BBB', 'AAA', 'CCC']
    assert ranks[0]['movement'] == 0  # already first by margin
    assert ranks[0]['power'] == 7.2


def test_form_can_leapfrog_season_margin():
    data = payload()
    data['children'][0]['standings']['entries'][0] = entry('AAA', 50, 32, '+5.9', 1, '10-0')
    ranks = st.power_rankings(st.parse_standings('nba', data))
    assert ranks[0]['abbrev'] == 'AAA' and ranks[0]['movement'] == 1


def test_playoff_picture_zones_and_gap():
    teams = [entry(f'T{i}', 60 - i * 3, 22 + i * 3, '+1.0', i) for i in range(1, 12)]
    conferences = st.parse_standings('nba', dict(children=[dict(name='Eastern Conference', standings=dict(entries=teams))]))
    zones = [t['zone'] for t in st.playoff_picture('nba', conferences)['conferences'][0]['teams']]
    assert zones == ['direct'] * 6 + ['play-in'] * 4 + ['out']
    seventh = st.playoff_picture('nba', conferences)['conferences'][0]['teams'][6]
    assert seventh['games_back_of_direct'] == 3.0


def test_get_standings_falls_back_to_previous_season(monkeypatch):
    empty = dict(children=[dict(name='Eastern Conference', standings=dict(entries=[entry('AAA', 0, 0, '0', 1)]))])
    seen = []
    def fake(url, ttl=0, **params):
        seen.append(params['season'])
        return empty if len(seen) == 1 else payload()
    monkeypatch.setattr(st, '_fetch_url', fake)
    result = st.get_standings('nba')
    assert result['fallback'] is True and seen[1] == seen[0] - 1 and result['power_rankings']
