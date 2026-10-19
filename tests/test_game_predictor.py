"""Pregame safety and feature correctness independent of remote data."""
import sys
from types import SimpleNamespace
import pandas as pd
import pytest
from utils.game_predictor import GamePredictor


@pytest.fixture(autouse=True)
def disable_external_model_artifact(monkeypatch):
    from utils import game_data
    monkeypatch.setattr(game_data, "load_game_team_history", lambda league: None)
    # Unit fixtures must be independent of whichever trained artifact is on disk.
    monkeypatch.setitem(sys.modules, "utils.game_model", SimpleNamespace(
        GameForecastModel=SimpleNamespace(load=lambda league: None)))


def logs():
    rows = []
    for index, day in enumerate(pd.date_range('2026-09-20', periods=5)):
        for team, opp, points, home in [('NYL', 'LVA', 80, True), ('LVA', 'NYL', 75, False)]:
            rows.append(dict(Game_ID=str(index), Player_ID=team, PLAYER_NAME=team+' Player',
                             MATCHUP=f'{team} {"vs." if home else "@"} {opp}',
                             _date=day, PTS=points, WL='W' if home else 'L', MIN=30,
                             AST=5, REB=7, STL=1, BLK=1))
    return pd.DataFrame(rows)


def test_target_date_rest_and_back_to_back():
    model = GamePredictor(pd.DataFrame(), logs(), league='wnba')
    model._set_cutoff('2026-09-25')
    assert model._get_rest_days('NYL') == 0
    model._set_cutoff('2026-09-28')
    assert model._get_rest_days('NYL') == 3


def test_future_results_and_full_season_statistics_cannot_change_forecast():
    frame = logs()
    before = GamePredictor(pd.DataFrame(), frame, league='wnba').predict_game('NYL', 'LVA', target_date='2026-09-25')
    future = frame.iloc[[0]].copy()
    future['_date'] = pd.Timestamp('2026-09-25')
    future['PTS'] = 1000
    future['Game_ID'] = 'future'
    defense = pd.DataFrame([dict(TEAM_ABBREVIATION='NYL', SEASON='2026', OPP_PTS=1000, PLUS_MINUS=1000)])
    after = GamePredictor(defense, pd.concat([frame, future]), league='wnba').predict_game('NYL', 'LVA', target_date='2026-09-25')
    assert before['predicted_total'] == after['predicted_total']
    assert before['predicted_spread'] == after['predicted_spread']
    assert after['data_quality']['teams']['NYL']['latest_game_date'] == '2026-09-24'


def test_player_game_duplicates_never_inflate_team_scores():
    frame = logs()
    model = GamePredictor(pd.DataFrame(), pd.concat([frame, frame.iloc[[0]]]), league='wnba')
    assert len(model._scores) == 10
    assert model._scores[model._scores.team == 'NYL'].team_pts.tolist() == [80]*5


def test_neutral_site_removes_home_advantage_and_fallback_is_explicit():
    model = GamePredictor(pd.DataFrame(), logs(), league='wnba')
    home = model.predict_game('NYL', 'LVA', target_date='2026-09-25')
    neutral = model.predict_game('NYL', 'LVA', target_date='2026-09-25', neutral=True)
    assert home['predicted_spread'] - neutral['predicted_spread'] == pytest.approx(2)
    assert neutral['model_metadata']['validated'] is False
    assert neutral['home_win_probability'] is None
    assert neutral['winner_confidence'] == 'LOW'


def test_injury_scenario_handles_questionable_and_already_absent_players():
    model = GamePredictor(pd.DataFrame(), logs(), league='wnba')
    baseline = model.predict_game('NYL', 'LVA', target_date='2026-09-25')
    injury = model.predict_game('NYL', 'LVA', target_date='2026-09-25', home_injuries=[dict(name='NYL Player', status='QUESTIONABLE')])
    assert injury['predicted_home_score'] < baseline['predicted_home_score']
    assert injury['predicted_away_score'] > baseline['predicted_away_score']
    assert injury['home_missing'][0]['adjustment_validated'] is False
    assert injury['model_metadata']['injury_scenario_applied']


def test_wnba_missing_history_uses_wnba_scoring_not_nba():
    model = GamePredictor(pd.DataFrame(), logs(), league='wnba')
    prediction = model.predict_game('TOR', 'POR', target_date='2026-09-25')
    assert prediction['predicted_total'] < 180
    assert prediction['data_quality']['teams']['TOR']['games_available'] == 0


def test_promoted_forecast_contract_probability_and_intervals(monkeypatch):
    received = {}
    def features(team_games, target_date, home, away, league, **kwargs):
        received.update(target_date=target_date, league=league, **kwargs)
        return {"test": 1}
    monkeypatch.setitem(sys.modules, "utils.game_features", SimpleNamespace(
        aggregate_team_games=lambda frame, league: frame,
        build_feature_vector=features))
    forecast = SimpleNamespace(predict=lambda vector: dict(margin=6, total=170,
        home_win_probability=.71, margin_sd=10, total_sd=12,
        model_name="validated_test", metrics={"margin_mae": 8},
        prediction_intervals={"margin_80": [-8, 20], "total_80": [150, 192]}))
    monkeypatch.setitem(sys.modules, "utils.game_model", SimpleNamespace(
        GameForecastModel=SimpleNamespace(load=lambda league: forecast)))
    model = GamePredictor(pd.DataFrame(), logs(), league='wnba')
    prediction = model.predict_game('NYL', 'LVA', target_date='2026-09-25', neutral=True, venue='Test arena')
    assert prediction['predicted_total'] == 170
    assert prediction['predicted_spread'] == 6
    assert prediction['home_win_probability'] == .71
    assert prediction['prediction_intervals']['coverage'] == .8
    assert prediction['model_metadata']['validated'] is True
    assert received['neutral'] and received['venue'] == 'Test arena'


def test_authoritative_team_boxscores_override_partial_player_totals(monkeypatch):
    from utils import game_data
    canonical = pd.DataFrame([
        dict(date=pd.Timestamp('2026-09-24'), team='NYL', opponent='LVA', home=True, PTS=92, margin=10),
        dict(date=pd.Timestamp('2026-09-24'), team='LVA', opponent='NYL', home=False, PTS=82, margin=-10),
        dict(date=pd.Timestamp('2026-09-25'), team='NYL', opponent='LVA', home=True, PTS=999, margin=900),
    ])
    monkeypatch.setattr(game_data, 'load_game_team_history', lambda league: canonical)
    model = GamePredictor(pd.DataFrame(), logs(), league='wnba')
    prediction = model.predict_game('NYL', 'LVA', target_date='2026-09-25')
    assert model._scores.team_pts.max() == 92
    assert model._scores.shape[0] == 2
    assert prediction['data_quality']['team_history_source'] == 'validated_team_boxscores'
    assert prediction['data_quality']['teams']['NYL']['latest_game_date'] == '2026-09-24'


def test_future_trained_artifact_cannot_predict_historical_date(monkeypatch):
    monkeypatch.setitem(sys.modules, 'utils.game_features', SimpleNamespace(
        aggregate_team_games=lambda frame, league: frame,
        build_feature_vector=lambda *args, **kwargs: {}))
    forecast = SimpleNamespace(predict=lambda vector: dict(margin=900, total=1000,
        trained_through='2026-10-01', model_name='future-model'))
    monkeypatch.setitem(sys.modules, 'utils.game_model', SimpleNamespace(
        GameForecastModel=SimpleNamespace(load=lambda league: forecast)))
    prediction = GamePredictor(pd.DataFrame(), logs(), league='wnba').predict_game(
        'NYL', 'LVA', target_date='2026-09-25')
    assert prediction['predicted_total'] < 200
    assert prediction['model_metadata']['validated'] is False
    assert prediction['home_win_probability'] is None
