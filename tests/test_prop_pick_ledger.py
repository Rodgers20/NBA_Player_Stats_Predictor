"""Published prop picks: first price kept, graded against the box score and the closing line."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from utils import prop_pick_ledger as ledger

NOW = datetime(2026, 10, 26, 16, 0, tzinfo=timezone.utc)   # noon ET, before a 7 PM ET tip
GAMES = {'games': [{'matchup': 'ATL @ HOU', 'status_state': 'pre', 'start_time': '2026-10-26T23:00Z'},
                   {'matchup': 'CHI @ BOS', 'status_state': 'in', 'start_time': '2026-10-26T15:00Z'}]}


def pick(**overrides):
    row = dict(player='Alperen Sengun', team='HOU', opponent='ATL', stat='REB', direction='Over', line=9.5,
               price=-110, book='FanDuel', ev=0.04, model_prob=0.55, market_prob=0.5, pick_type='pick',
               game_matchup='ATL @ HOU')
    return {**row, **overrides}


def board(*props, date='2026-10-26'):
    return {'target_date': date, 'props': list(props)}


def logs(*rows):
    return pd.DataFrame([dict(PLAYER_NAME=p, _date=pd.Timestamp(d), PTS=pts, REB=reb, AST=ast) for p, d, pts, reb, ast in rows])


def test_records_only_priced_picks_before_tip(tmp_path):
    path = tmp_path / 'ledger.json'
    added = ledger.record_picks('nba', board(pick(), pick(player='Lean', pick_type='lean'), pick(player='No price', price=None),
                                             pick(player='Started', game_matchup='CHI @ BOS')), GAMES, path=path, now=NOW)
    assert added == 1
    assert [p['player'] for p in ledger.load('nba', path)['picks']] == ['Alperen Sengun']


def test_first_published_price_is_kept(tmp_path):
    path = tmp_path / 'ledger.json'
    ledger.record_picks('nba', board(pick(price=-110)), GAMES, path=path, now=NOW)
    assert ledger.record_picks('nba', board(pick(price=+120, line=10.5)), GAMES, path=path, now=NOW) == 0
    saved = ledger.load('nba', path)['picks'][0]
    assert (saved['price'], saved['line']) == (-110, 9.5)


def test_grades_result_profit_and_clv(tmp_path):
    path = tmp_path / 'ledger.json'
    ledger.record_picks('nba', board(pick(), pick(stat='PTS+REB', line=30.5, price=+150), pick(stat='AST', line=5, direction='Under')),
                        GAMES, path=path, now=NOW)
    history = logs(('Alperen Sengun', '2026-10-26', 21, 11, 5))
    close = lambda *args: {'close_line': 10.5, 'close_price': -120, 'clv_pct': 6.8}
    assert ledger.grade_picks('nba', history, path=path, today=pd.Timestamp('2026-10-27'), closing=close) == 3
    reb, combo, ast = ledger.load('nba', path)['picks']
    assert (reb['result'], reb['actual'], reb['units']) == ('win', 11, pytest.approx(100 / 110))
    assert (combo['result'], combo['units']) == ('win', 1.5)
    assert (ast['result'], ast['units']) == ('push', 0)
    assert reb['clv_pct'] == 6.8


def test_waits_for_box_score_then_voids_a_player_who_never_played(tmp_path):
    path = tmp_path / 'ledger.json'
    ledger.record_picks('nba', board(pick()), GAMES, path=path, now=NOW)
    no_rows = logs(('Someone Else', '2026-10-26', 10, 2, 2))
    closing = lambda *args: {'close_line': None, 'close_price': None, 'clv_pct': None}
    assert ledger.grade_picks('nba', no_rows, path=path, today=pd.Timestamp('2026-10-27'), closing=closing) == 0
    assert ledger.load('nba', path)['picks'][0]['result'] is None
    ledger.grade_picks('nba', no_rows, path=path, today=pd.Timestamp('2026-10-30'), closing=closing)
    assert ledger.load('nba', path)['picks'][0]['result'] == 'void'


def test_summary_reports_hit_rate_roi_and_clv():
    picks = [dict(date='2026-10-26', stat='REB', result='win', units=0.91, clv_pct=3.0),
             dict(date='2026-10-26', stat='REB', result='loss', units=-1, clv_pct=-1.0),
             dict(date='2026-10-26', stat='PTS', result='push', units=0, clv_pct=None),
             dict(date='2026-10-27', stat='PTS', result=None, units=None, clv_pct=None)]
    summary = ledger.summary({'league': 'nba', 'picks': picks})
    assert (summary['hit'], summary['miss'], summary['pushes'], summary['total'], summary['pending']) == (1, 1, 1, 2, 1)
    assert summary['pct'] == 50.0
    assert summary['roi_pct'] == -3.0
    assert summary['clv'] == {'n': 2, 'avg_pct': 1.0, 'beat_pct': 50.0}
    assert summary['by_stat']['REB'] == {'hit': 1, 'total': 2, 'pct': 50.0}


def test_empty_ledger_summary_is_zero():
    summary = ledger.summary({'league': 'nba', 'picks': []})
    assert (summary['total'], summary['pct'], summary['clv']['n']) == (0, 0.0, 0)
