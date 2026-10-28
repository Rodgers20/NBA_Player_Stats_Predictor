"""Player history must hold only real-season games the player actually played."""
import pandas as pd

from api.data import played_games


def _logs(**columns):
    return pd.DataFrame(columns)


def test_drops_preseason_and_all_star_rows_keeps_regular_season_and_playoffs():
    logs = _logs(SEASON_ID=['12026', '22026', '32026', '42026', '52026'], MIN=[18, 30, 20, 38, 35],
                 PTS=[9, 35, 40, 30, 22], REB=[3, 9, 5, 12, 7], AST=[1, 5, 4, 6, 3])
    assert list(played_games(logs)['SEASON_ID']) == ['22026', '42026', '52026']


def test_drops_did_not_play_rows_but_keeps_a_played_game_missing_its_minutes():
    logs = _logs(SEASON_ID=['22026'] * 4, MIN=[None, 0, 31, None],
                 PTS=[0, 0, 24, 11], REB=[0, 0, 8, 2], AST=[0, 0, 5, 1])
    kept = played_games(logs)
    assert list(kept['PTS']) == [24, 11]   # the 0-minute and blank-minute zero rows are DNPs


def test_works_without_season_or_minutes_columns():
    logs = _logs(PTS=[10, 0], REB=[2, 0], AST=[1, 0])
    assert len(played_games(logs)) == 2
