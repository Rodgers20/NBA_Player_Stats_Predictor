import numpy as np
import pandas as pd
import pytest

from models.predictor import StatPredictor
from models.validation import dated_frame, date_partitions, expanding_folds, safe_features
from utils.feature_engineering import engineer_features
from utils.pregame_features import next_game_features, prefer_identified_games
from utils.market_evaluation import evaluate_market
from utils.prop_calculator import calculate_hit_probability, calculate_historical_hit_rate


def logs(n=90):
    rows = []
    for player in ('A', 'B'):
        for i, day in enumerate(pd.date_range('2025-01-01', periods=n)):
            rows.append(dict(PLAYER_NAME=player, GAME_DATE=str(day.date()), SEASON='2024-25',
                             MATCHUP='BOS vs. MIA', PTS=10 + i % 8, AST=2 + i % 3,
                             REB=4 + i % 4, MIN=5 if i % 8 == 0 else 30,
                             FGA=10, FG_PCT=.5, FG3A=3, FG3_PCT=.3, PF=i % 6,
                             PLAYER_ID=42, TEAM_ID=99))
    return pd.DataFrame(rows)


def test_future_outcomes_do_not_change_pregame_features():
    original = logs()
    changed = original.copy()
    changed.loc[changed.GAME_DATE >= '2025-02-01', ['PTS', 'AST', 'REB', 'MIN']] = 999
    a, b = engineer_features(original), engineer_features(changed)
    cols = safe_features(a)
    pd.testing.assert_frame_equal(a.loc[a.GAME_DATE <= '2025-02-01', cols],
                                  b.loc[b.GAME_DATE <= '2025-02-01', cols])
    assert 'PF' not in cols and 'TEAM_ID' not in cols and 'MIN' not in cols


def test_date_partitions_and_folds_never_share_a_day():
    engineered = engineer_features(logs())
    frame = dated_frame(engineered, safe_features(engineered), 'PTS')
    dev, cal, test = date_partitions(frame)
    assert dev._validation_date.max() < cal._validation_date.min()
    assert cal._validation_date.max() < test._validation_date.min()
    for a, b in expanding_folds(dev):
        assert dev.iloc[a]._validation_date.max() < dev.iloc[b]._validation_date.min()
    assert (test.MIN == 5).any(), 'Do not hide low-minute outcomes'


def test_next_game_features_match_training_row_and_include_latest_game():
    raw = logs(40).query('PLAYER_NAME == "A"')
    engineered = engineer_features(raw)
    expected = engineered.iloc[-1]
    actual = next_game_features(raw.iloc[:-1], expected.GAME_DATE, True)
    for feature in safe_features(engineered):
        assert actual[feature] == pytest.approx(expected[feature]), feature
    assert actual['rolling_avg_pts_5'] == pytest.approx(raw.PTS.iloc[-6:-1].mean())


def test_duplicates_cannot_inflate_history():
    raw = logs(40)
    pd.testing.assert_frame_equal(engineer_features(raw), engineer_features(pd.concat([raw, raw])))
    duplicate = raw.iloc[[0]].copy()
    duplicate['PTS'] = 100
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        engineer_features(pd.concat([raw, duplicate]))


def test_identified_box_score_supersedes_partial_live_duplicate():
    raw = logs(40)
    raw['Game_ID'] = np.arange(len(raw)) + 1
    partial = raw.iloc[[0]].copy()
    partial['Game_ID'] = np.nan
    partial['PTS'] = 999  # A stale live-feed score must not displace the full box score.
    canonical = prefer_identified_games(pd.concat([partial, raw], ignore_index=True))
    assert len(canonical) == len(raw)
    assert canonical.PTS.max() < 999
    conflict = raw.iloc[[0]].copy()
    conflict['PTS'] = 999
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        engineer_features(prefer_identified_games(pd.concat([conflict, raw], ignore_index=True)))


def test_training_uses_newer_cache_raw_columns_only(tmp_path, monkeypatch):
    from scripts import train_improved_models as trainer
    old = logs(35)
    new = logs(40)
    new['rolling_avg_pts_5'] = 999  # Cached features must be recomputed, never trusted.
    old.to_csv(tmp_path / 'player_game_logs.csv', index=False)
    (tmp_path / 'engineered_data.parquet').touch()
    monkeypatch.setattr(trainer, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(trainer.pd, 'read_parquet', lambda path: new)
    loaded = trainer.load_all_data()
    assert len(loaded['game_logs']) == 80
    assert 'rolling_avg_pts_5' not in loaded['game_logs']
    assert loaded['data_source'].endswith('engineered_data.parquet')


def test_invalid_dates_and_unsafe_weights_fail():
    raw = logs()
    raw.loc[0, 'GAME_DATE'] = 'unknown'
    with pytest.raises(ValueError, match='GAME_DATE'):
        engineer_features(raw)
    with pytest.raises(ValueError, match='Outcome-based'):
        StatPredictor('ridge').train(logs(), weight_column='MIN')


def test_market_prices_pushes_and_missing_data():
    residuals = np.zeros(100)
    over = evaluate_market(10, 10, 150, residuals, 'over')
    under = evaluate_market(10, 10, -110, residuals, 'under')
    assert over['implied_prob'] == .4
    assert over['push_prob'] > .98
    assert over['model_prob'] + under['model_prob'] + over['push_prob'] == pytest.approx(1)
    assert over['ev'] == pytest.approx(over['model_prob'] * 1.5 - under['model_prob'])
    assert evaluate_market(10, 9.5, None, residuals) is None
    assert evaluate_market(10, 9.5, 0, residuals) is None
    assert evaluate_market(10, 9.5, -110, None) is None
    assert calculate_historical_hit_rate(pd.Series([10]), 10, 'over') == 0
    assert calculate_hit_probability(4, 4, 2, 'under', 'AST') < calculate_hit_probability(4, 4.5, 2, 'under', 'AST')


def test_validated_model_roundtrip_and_missing_feature_rejection(tmp_path):
    model = StatPredictor('ridge')
    frame = engineer_features(logs())
    metrics = model.train(frame)
    assert metrics['probability_diagnostic']['market_calibrated'] is False
    assert metrics['test_samples'] > 0
    path = tmp_path / 'model.pkl'
    model.save(path)
    loaded = StatPredictor.load(path)
    row = next_game_features(logs().query('PLAYER_NAME == "A"'), '2025-04-01', True)
    assert loaded.predict(row) == model.predict(row)
    np.testing.assert_equal(loaded.calibration_residuals, model.calibration_residuals)
    with pytest.raises(ValueError, match='Missing pregame'):
        loaded.predict({})
