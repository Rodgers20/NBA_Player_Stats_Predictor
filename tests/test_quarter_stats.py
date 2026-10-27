import pandas as pd

from utils import quarter_stats, trends


def _history():
    return pd.DataFrame({
        'Player_ID': [1, 1, 2, 1],
        'Game_ID': [22600001, 22600002, 22600002, 22500900],
        'PLAYER_NAME': ['A', 'A', 'B', 'A'],
        'PTS': [20, 25, 4, 18],
    })


def _q1():
    # stats.nba.com spells the ids as zero-padded strings; B played the game but not the first quarter.
    return pd.DataFrame({'PLAYER_ID': [1, 1], 'GAME_ID': ['0022600001', '0022600002'],
                         'PTS': [7, 9], 'REB': [2, 1], 'AST': [1, 3], 'FG3M': [1, 0]})


def test_first_quarter_lines_join_onto_each_game():
    joined = quarter_stats.with_q1(_history(), _q1())
    assert list(joined['PTS_1Q'])[:2] == [7, 9]
    assert list(joined['AST_1Q'])[:2] == [1, 3]
    assert list(joined['PTS']) == [20, 25, 4, 18], 'full-game columns are untouched'
    assert 'PTS_1Q' not in _history(), 'the input frame is not mutated'


def test_covered_game_without_a_first_quarter_line_is_zero_and_uncovered_game_is_empty():
    joined = quarter_stats.with_q1(_history(), _q1())
    assert joined['PTS_1Q'].iloc[2] == 0, 'B sat the first quarter of a covered game'
    assert pd.isna(joined['PTS_1Q'].iloc[3]), 'last season was not fetched, so it is unknown, not zero'


def test_no_first_quarter_data_leaves_history_as_it_was():
    joined = quarter_stats.with_q1(_history(), pd.DataFrame())
    assert all(pd.isna(joined['PTS_1Q']))


def test_first_quarter_trend_cards_use_the_1q_columns():
    dates = pd.date_range('2026-10-01', periods=10)
    history = pd.DataFrame({'PLAYER_NAME': 'A', '_date': dates, 'MIN': 34, 'PTS': 25, 'REB': 8, 'AST': 6, 'FG3M': 2,
                            'STL': 1, 'BLK': 1, 'PTS_1Q': [8, 9, 7, 8, 10, 9, 8, 7, 9, 8], 'REB_1Q': 2, 'AST_1Q': 1, 'FG3M_1Q': 0})
    cards = trends.build_trends(history, today=pd.Timestamp('2026-10-12'), stats=quarter_stats.Q1_STATS)
    assert cards and {card['stat'] for card in cards} <= set(quarter_stats.Q1_STATS)
    points = next(card for card in cards if card['stat'] == 'PTS_1Q')
    assert points['values'] == [8, 9, 7, 8, 10, 9, 8, 7, 9, 8]
    assert '1st-quarter points' in points['text']
