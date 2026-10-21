import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.game_features import (aggregate_team_games, build_game_features,
    build_feature_vector, build_matchup_features, normalize_team_games, FEATURE_COLUMNS)


def logs():
    rows = []
    for game, day in [('1', '2026-10-01'), ('2', '2026-10-03'), ('3', '2026-10-05')]:
        for team, matchup, points in [('NY', 'NY vs. BOS', 20), ('BOS', 'BOS @ NY', 19)]:
            for player in range(5):
                rows.append(dict(Game_ID=game, GAME_DATE=day, Player_ID=f'{team}{player}',
                    TEAM_ABBREVIATION=team, MATCHUP=matchup, SEASON_ID='22026', MIN=48,
                    PTS=points, FGA=16, FGM=8, FG3M=2, FTA=4, FTM=2, OREB=2, DREB=5, TOV=2))
    return pd.DataFrame(rows)


def test_future_outcomes_cannot_change_prior_features():
    original = logs()
    changed = original.copy()
    changed.loc[changed.Game_ID == '3', ['PTS', 'FGM', 'TOV']] = 999
    a, b = build_game_features(original), build_game_features(changed)
    assert_frame_equal(a.loc[:1], b.loc[:1])
    assert_frame_equal(a[list(FEATURE_COLUMNS)], b[list(FEATURE_COLUMNS)])


def test_pairs_and_dedupe_and_complete_minutes():
    data = logs()
    assert len(aggregate_team_games(pd.concat([data, data.iloc[:1]]))) == 6
    assert len(build_game_features(data)) == 3
    data.loc[(data.Game_ID == '2') & (data.TEAM_ABBREVIATION == 'BOS'), 'MIN'] = 1
    result = aggregate_team_games(data)
    assert set(result.game_id) == {'1', '3'}
    assert result.groupby('game_id').size().eq(2).all()


def test_future_and_historical_vectors_match_and_same_day_excluded():
    games = aggregate_team_games(logs())
    batch = build_matchup_features(games)
    vector = build_feature_vector(games, '2026-10-03', 'NYK', 'BOS')
    for key, value in vector.items():
        actual = batch.iloc[1][key]
        assert (pd.isna(value) and pd.isna(actual)) or value == actual
    assert vector['home_rest_days'] == 1
    assert vector['home_games_last4'] == 1
    assert vector['home_offense_5'] == 100 * 100 / (80 + .44 * 20 - 10 + 10)


def test_overtime_pace_and_missing_stats():
    data = logs()
    data.loc[data.Game_ID == '1', 'MIN'] = 53
    data.loc[0, 'FTA'] = np.nan
    games = aggregate_team_games(data)
    first = games[games.game_id == '1']
    assert first.offense.isna().all()
    assert first.FTA.isna().sum() == 1
    second = games[games.game_id == '2']
    assert np.isclose(second.iloc[0].pace, 88.8)


def test_preseason_excluded_and_team_input_supported():
    data = logs()
    data.loc[data.Game_ID == '1', 'SEASON_ID'] = '12026'
    assert set(aggregate_team_games(data).game_id) == {'2', '3'}
    sums = logs().groupby(['Game_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'SEASON_ID']).sum(numeric_only=True).reset_index()
    assert len(normalize_team_games(sums)) == 6


def test_dnp_rows_and_missing_team_abbreviation_use_matchup():
    data = logs()
    data['TEAM_ABBREVIATION'] = np.nan
    dnp = data.iloc[:1].copy()
    dnp['Player_ID'] = 'DNP'
    dnp[['MIN', 'PTS']] = [np.nan, 0]
    result = aggregate_team_games(pd.concat([data, dnp]))
    assert len(result) == 6
    assert set(result.team) == {'NYK', 'BOS'}
    data.loc[0, 'MIN'] = np.nan
    assert set(aggregate_team_games(data).game_id) == {'2', '3'}


def test_overtime_normalization_and_nonreciprocal_rejection():
    data = logs()
    data['MIN'] = data['MIN'].astype(object)
    data.loc[data.Game_ID == '1', 'MIN'] = '53:00'
    games = aggregate_team_games(data)
    assert np.isclose(games[games.game_id == '1'].iloc[0].pace, 88.8 * 48 / 53)
    data.loc[(data.Game_ID == '2') & (data.TEAM_ABBREVIATION == 'BOS'), 'MATCHUP'] = 'BOS @ LAL'
    assert set(aggregate_team_games(data).game_id) == {'1', '3'}


def test_same_day_outcomes_and_neutral_unknown_travel():
    data = logs()
    data.loc[data.Game_ID == '2', 'GAME_DATE'] = '2026-10-01'
    games = aggregate_team_games(data)
    vector = build_feature_vector(games, '2026-10-01', 'NYK', 'BOS', neutral=True)
    assert vector['home_history_games'] == 0
    assert vector['home_advantage'] == 0
    assert vector['home_travel_missing'] == 1
    assert set(vector) == set(FEATURE_COLUMNS)
