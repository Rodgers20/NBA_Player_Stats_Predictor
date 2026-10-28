import pandas as pd

from utils.game_context import annotate, build_league_context


def _rows(*games):
    """games: (name, game_id, date, matchup, minutes, result)"""
    frame = pd.DataFrame(games, columns=['PLAYER_NAME', 'Game_ID', '_date', 'MATCHUP', 'MIN', 'WL'])
    frame['_date'] = pd.to_datetime(frame['_date'])
    frame['SEASON'] = '2025-26'
    return frame


def _teams(*rows):
    """rows: (game_id, team, points, opponent_points)"""
    return pd.DataFrame(rows, columns=['GAME_ID', 'TEAM_ABBREVIATION', 'PTS', 'OPP_PTS'])


# Star plays every game; Big (a rotation player) misses game 3; Bench never plays enough minutes to matter.
HISTORY = _rows(
    ('Star', 1, '2025-11-01', 'AAA vs. BBB', 30.0, 'W'),
    ('Star', 2, '2025-11-02', 'AAA @ CCC', 32.0, 'L'),
    ('Star', 3, '2025-11-05', 'AAA vs. DDD', 28.0, 'W'),
    ('Big', 1, '2025-11-01', 'AAA vs. BBB', 25.0, 'W'),
    ('Big', 2, '2025-11-02', 'AAA @ CCC', 24.0, 'L'),
    ('Big', 4, '2025-11-06', 'AAA vs. EEE', 26.0, 'W'),
    ('Big', 5, '2025-11-07', 'AAA vs. FFF', 27.0, 'W'),
    ('Big', 6, '2025-11-08', 'AAA vs. GGG', 22.0, 'L'),
    ('Bench', 1, '2025-11-01', 'AAA vs. BBB', 6.0, 'W'),
    ('Bench', 2, '2025-11-02', 'AAA @ CCC', 5.0, 'L'),
    ('Bench', 3, '2025-11-05', 'AAA vs. DDD', 4.0, 'W'),
)
TEAMS = _teams((1, 'AAA', 110, 100), (2, 'AAA', 98, 104), (3, 'AAA', 120, 90))


def _star_games():
    context = build_league_context(HISTORY, TEAMS)
    star = HISTORY[HISTORY['PLAYER_NAME'] == 'Star'].sort_values('_date')
    return annotate(star, context)


def test_minutes_and_result_come_from_the_log():
    games = _star_games()
    assert [game['minutes'] for game in games] == [30.0, 32.0, 28.0]
    assert [game['won'] for game in games] == [True, False, True]


def test_margin_is_team_points_minus_opponent_points():
    assert [game['margin'] for game in _star_games()] == [10, -6, 30]


def test_margin_is_unknown_when_the_game_has_no_team_score():
    context = build_league_context(HISTORY, _teams((1, 'AAA', 110, 100)))
    star = HISTORY[HISTORY['PLAYER_NAME'] == 'Star'].sort_values('_date')
    assert [game['margin'] for game in annotate(star, context)] == [10, None, None]


def test_player_logs_that_say_san_match_team_history_that_says_sas():
    history = _rows(('Spur', 1, '2025-11-01', 'SAN vs. BBB', 30.0, 'W'))
    context = build_league_context(history, _teams((1, 'SAS', 101, 99)))
    assert annotate(history, context)[0]['margin'] == 2


def test_rest_days_count_the_days_between_games_and_skip_the_season_gap():
    games = _star_games()
    assert [game['rest'] for game in games] == [None, 0, 2]   # back to back, then two full days
    gap = _rows(('Star', 1, '2025-04-01', 'AAA vs. BBB', 30.0, 'W'), ('Star', 2, '2025-10-25', 'AAA vs. BBB', 30.0, 'W'))
    assert annotate(gap, build_league_context(gap, TEAMS))[1]['rest'] is None


def test_out_lists_rotation_teammates_who_missed_the_game():
    games = _star_games()
    assert [game['out'] for game in games] == [[], [], ['Big']]   # Big missed game 3; Bench is not a rotation player


def test_a_player_is_never_out_for_himself():
    big = HISTORY[HISTORY['PLAYER_NAME'] == 'Big'].sort_values('_date')
    context = build_league_context(HISTORY, TEAMS)
    assert all('Big' not in game['out'] for game in annotate(big, context))


def test_out_is_capped_and_ordered_by_minutes():
    names = [f'P{i}' for i in range(10)]
    rows = [(name, 1, '2025-11-01', 'AAA vs. BBB', 40.0 - i, 'W') for i, name in enumerate(names)]
    rows += [(name, game, f'2025-11-0{game}', 'AAA vs. BBB', 40.0 - i, 'W') for i, name in enumerate(names) for game in (2, 3, 4, 5, 6)]
    rows += [('Star', 7, '2025-11-09', 'AAA vs. BBB', 30.0, 'W')]    # nobody else plays game 7
    history = _rows(*rows)
    star = history[history['PLAYER_NAME'] == 'Star']
    out = annotate(star, build_league_context(history, TEAMS))[0]['out']
    assert out == names[:6]
