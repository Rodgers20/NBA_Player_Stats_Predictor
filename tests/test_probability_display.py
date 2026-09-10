"""Render changed callbacks without running app startup/network schedulers."""
import ast
from pathlib import Path
from dash import html
from utils.wnba_props import _build_prop, SYNTHETIC_BOOKMAKER
import pandas as pd


def callback_function(name):
    source = ast.parse(Path('dashboard/app.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {'html': html}
    exec(compile(module, 'dashboard/app.py', 'exec'), namespace)
    return namespace[name]


def test_game_confidence_category_is_not_a_fabricated_percentage():
    component = callback_function('update_game_panel')(0, [{'home': 'BOS', 'away': 'MIA',
        'winner_conf': 'HIGH', 'home_win_pct': None, 'winner_pick': 'BOS'}])
    rendered = str(component)
    assert 'Probability not calibrated' in rendered
    assert '(85%)' not in rendered and 'None%' not in rendered


def test_unpriced_wnba_prop_renders_na_instead_of_fake_ev():
    prop = _build_prop(player_name='A', team='BOS', stat='PTS', line=15.5, projected=20,
        actual_series=pd.Series([20] * 10), over_price=None, under_price=None,
        bookmaker=SYNTHETIC_BOOKMAKER, recent_n=10)
    rendered = str(callback_function('_wnba_props_row')(prop))
    assert 'N/A' in rendered
    assert 'EST. PROB.' in rendered
