import numpy as np
import pandas as pd

from utils import props_cache
from utils.wnba_props import _build_prop, SYNTHETIC_BOOKMAKER


class Model:
    calibration_residuals = np.linspace(-5, 5, 100)

    def predict_player_game(self, player, history, **kwargs):
        assert kwargs['is_home'] is True
        return {'predicted_pts': 20.0}


def test_nba_props_use_one_actual_line_and_keep_low_minute_games(monkeypatch):
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=20)
    history = pd.DataFrame({'PLAYER_NAME': ['A'] * 20, '_date': dates,
                            'GAME_DATE': dates.strftime('%Y-%m-%d'), 'MATCHUP': ['BOS vs. MIA'] * 20,
                            'MIN': [30] * 19 + [5], 'PTS': [20] * 19 + [0]})
    positions = pd.DataFrame({'PLAYER_NAME': ['A'], 'TEAM_ABBREVIATION': ['BOS'], 'POSITION': ['G']})
    info = {'has_todays_games': True, 'team_to_opponent': {'BOS': 'MIA'},
            'teams_home_away': {'BOS': 'home'}}
    quote = {'A': {'PTS': {'line': 17.5, 'over_price': 150, 'bookmaker': 'Test'}}}
    monkeypatch.setattr(props_cache, 'get_live_odds', lambda: quote)
    results = props_cache._compute_main_page_props(history, positions, pd.DataFrame(), info,
        {}, ['A'], get_predictor_fn=lambda stat: Model() if stat == 'PTS' else None)
    assert len(results) == 1
    prop = results[0]
    assert prop['line'] == prop['book_line'] == prop['live_line'] == 17.5
    assert prop['implied_prob'] == .4
    assert prop['hit_rate_l5'] == .8
    assert prop['l5_values'][0] == 0
    assert prop['chart_windows']['l5']['values'][-1] == 0
    assert prop['confidence'] == 'LOW' and not prop['is_lock']
    monkeypatch.setattr(props_cache, 'get_live_odds', lambda: {})
    assert props_cache._compute_main_page_props(history, positions, pd.DataFrame(), info,
        {}, ['A'], get_predictor_fn=lambda stat: Model()) == []


def test_wnba_synthetic_props_have_no_price_or_ev():
    prop = _build_prop(player_name='A', team='BOS', stat='PTS', line=15.5, projected=20,
                      actual_series=pd.Series([20] * 10), over_price=None, under_price=None,
                      bookmaker=SYNTHETIC_BOOKMAKER, recent_n=10)
    assert prop.ev is None and not prop.has_live_odds
    assert prop.over_price is None and prop.under_price is None
    assert prop.hit_prob < 1


def test_wnba_prices_both_sides_with_residuals():
    prop = _build_prop(player_name='A', team='BOS', stat='PTS', line=19.5, projected=20,
                      actual_series=pd.Series([20] * 10), over_price=-1000, under_price=200,
                      bookmaker='Test', recent_n=10, residuals=np.linspace(-5, 5, 100))
    assert prop.pick == 'UNDER'  # Better price can outweigh the mean being above the line.
    assert prop.ev > 0
    assert prop.has_live_odds
