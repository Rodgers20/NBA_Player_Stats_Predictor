import pandas as pd

from api.routes import players

STATS = dict(PTS=20, REB=5, AST=4, FG3M=2, STL=1, BLK=0)


def _frame():
    rows = []
    for name, games in {'Test Player': (1, 2, 3), 'Big Man': (1, 2, 4, 5, 6)}.items():
        for game in games:
            rows.append(dict(PLAYER_NAME=name, Game_ID=game, _date=pd.Timestamp(f'2025-11-0{game}'), GAME_DATE=f'2025-11-0{game}',
                             MATCHUP='AAA vs. BBB', MIN=30.0 if name == 'Test Player' else 25.0, WL='W' if game % 2 else 'L',
                             SEASON='2025-26', TEAM_ABBREVIATION='AAA', **STATS))
    return pd.DataFrame(rows)


def _teams():
    return pd.DataFrame([(1, 'AAA', 110, 100), (2, 'AAA', 90, 99), (3, 'AAA', 101, 100)],
                        columns=['GAME_ID', 'TEAM_ABBREVIATION', 'PTS', 'OPP_PTS'])


def _patch(monkeypatch, frame=None, teams=None):
    frame = _frame() if frame is None else frame
    monkeypatch.setattr(players, 'history', lambda league: frame)
    monkeypatch.setattr(players, '_team_history', lambda league: teams)


def test_series_rows_carry_minutes_margin_result_rest_and_absent_teammates(monkeypatch):
    _patch(monkeypatch, teams=_teams())
    rows = players.get_player_series('Test Player', games=200, league='nba')['games']
    assert [row['minutes'] for row in rows] == [30.0, 30.0, 30.0]
    assert [row['margin'] for row in rows] == [10, -9, 1]
    assert [row['won'] for row in rows] == [True, False, True]
    assert [row.get('rest') for row in rows] == [None, 0, 0]
    assert [row.get('out') for row in rows] == [None, None, ['Big Man']]   # Big Man sat out game 3


def test_chart_rows_carry_the_same_context(monkeypatch):
    _patch(monkeypatch, teams=_teams())
    rows = players.get_player_chart_data('Test Player', stat='PTS', games=200, league='nba', line=None)['games']
    assert [row['margin'] for row in rows] == [10, -9, 1]
    assert rows[2]['out'] == ['Big Man']


def test_a_shorter_tail_keeps_context_aligned_with_its_games(monkeypatch):
    _patch(monkeypatch, teams=_teams())
    series = players.get_player_series('Test Player', games=2, league='nba')['games']
    chart = players.get_player_chart_data('Test Player', stat='PTS', games=2, league='nba', line=None)['games']
    for rows in (series, chart):
        assert [row['date'] for row in rows] == ['2025-11-02', '2025-11-03']
        assert [row['margin'] for row in rows] == [-9, 1]
        assert rows[-1]['out'] == ['Big Man']


def test_missing_values_are_omitted_rather_than_sent_as_null(monkeypatch):
    _patch(monkeypatch, teams=None)
    rows = players.get_player_series('Test Player', games=200, league='nba')['games']
    assert all('margin' not in row for row in rows)
    assert all('out' not in row for row in rows[:2])
    assert rows[0]['minutes'] == 30.0


def test_logs_without_the_context_columns_still_serve_the_basic_series(monkeypatch):
    plain = _frame().drop(columns=['Game_ID', 'MIN', 'WL'])
    _patch(monkeypatch, frame=plain, teams=None)
    rows = players.get_player_series('Test Player', games=200, league='nba')['games']
    assert len(rows) == 3 and all('minutes' not in row and 'margin' not in row for row in rows)
