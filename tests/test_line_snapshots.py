import pytest
from fastapi.testclient import TestClient
from api.main import app
from utils import line_snapshots, odds_budget, odds_store, personal_bets, prediction_tracker

client = TestClient(app)


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(odds_budget, 'DB_PATH', tmp_path / 'snap.sqlite3')


def quote(*books):
    return {'A': {'PTS': {'line': books[0][1], 'over_price': books[0][2], 'under_price': books[0][3],
                          'bookmaker': books[0][0],
                          'books': [dict(book=b, line=l, over_price=o, under_price=u) for b, l, o, u in books]}}}


def test_save_appends_snapshots_and_dedupes_identical_rows():
    odds_store.save('nba', quote(('FD', 20.5, -110, -110), ('DK', 21.5, -105, -125)))
    odds_store.save('nba', quote(('FD', 20.5, -110, -110), ('DK', 21.5, -105, -125)))   # unchanged
    odds_store.save('nba', quote(('FD', 21.5, -110, -110), ('DK', 21.5, -105, -125)))   # FD moved
    rows = line_snapshots.history('nba', 'A', 'PTS')
    assert [(r['book'], r['line']) for r in rows if r['book'] == 'FD'] == [('FD', 20.5), ('FD', 21.5)]
    assert len([r for r in rows if r['book'] == 'DK']) == 1
    assert [r['ts'] for r in rows] == sorted(r['ts'] for r in rows)
    assert line_snapshots.open_lines('nba')[('A', 'PTS', 'FD')] == 20.5


def test_legacy_single_book_quote_is_snapshotted():
    odds_store.save('wnba', {'B': {'REB': {'line': 5.5, 'over_price': -120, 'under_price': 100, 'bookmaker': 'FD'}}})
    assert line_snapshots.history('wnba', 'B', 'REB')[0]['book'] == 'FD'
    assert line_snapshots.history('nba', 'B', 'REB') == []


def test_old_snapshots_are_pruned():
    with odds_budget.connection() as database:
        line_snapshots.record(database, 'nba', quote(('FD', 20.5, -110, -110)), now=1_000)
        line_snapshots.record(database, 'nba', quote(('FD', 22.5, -110, -110)), now=1_000 + 31 * 86400)
    assert [r['line'] for r in line_snapshots.history('nba', 'A', 'PTS')] == [22.5]


def test_line_history_endpoint_ascending_and_validated():
    odds_store.save('nba', quote(('FD', 20.5, -110, -110)))
    odds_store.save('nba', quote(('FD', 21.5, -110, -110)))
    body = client.get('/api/props/line-history?league=nba&player=A&stat=PTS').json()
    assert body['player'] == 'A' and [s['line'] for s in body['snapshots']] == [20.5, 21.5]
    assert client.get('/api/props/line-history?league=nba&player=Nobody&stat=PTS').json()['snapshots'] == []
    assert client.get('/api/props/line-history?league=nba&stat=PTS').status_code == 422


def test_bet_clv_from_snapshots_and_explicit_close():
    odds_store.save('nba', quote(('FD', 21.5, -130, 110)))
    bet = dict(league='nba', player='A', stat='PTS', side='Over', line=20.5, price=-110, book='FD',
               game_date='2999-01-01')
    out = personal_bets.bet_clv(bet)
    assert out['close_line'] == 21.5 and out['clv_pct'] > 0
    explicit = personal_bets.bet_clv(dict(bet, close_line=20.5, close_price=-110))
    assert explicit['clv_pct'] == 0
    assert personal_bets.bet_clv(dict(bet, player='Nobody'))['clv_pct'] is None


def test_add_bet_accepts_and_validates_close_fields():
    base = dict(mode='paper', league='nba', player='A', game_date='2026-01-01', stat='PTS', side='Over',
                line=20.5, price=-110, book='FD', stake=10)
    identifier = personal_bets.add_bet(dict(base, close_line=21.5, close_price=-125), 't1')
    saved = next(b for b in personal_bets.list_bets() if b['id'] == identifier)
    assert saved['close_line'] == 21.5 and saved['close_price'] == -125.0
    with pytest.raises(ValueError):
        personal_bets.add_bet(dict(base, close_price=5), 't2')
    with pytest.raises(ValueError):
        personal_bets.add_bet(dict(base, close_line='abc'), 't3')


def test_grading_hook_fills_close_fields(monkeypatch):
    odds_store.save('nba', quote(('FD', 21.5, -130, 110)))
    prop = dict(player='A', stat='PTS', direction='Under', line=20.5)
    prediction_tracker._fill_closing_line(prop, '2999-01-01')
    assert (prop['close_line'], prop['close_price']) == (21.5, 110)
    missing = dict(player='Nobody', stat='PTS', direction='Over', line=1.5)
    prediction_tracker._fill_closing_line(missing, '2999-01-01')
    assert 'close_line' not in missing


def test_malformed_book_entry_still_saves_quote_and_valid_snapshots():
    q = quote(('FD', 20.5, -110, -110))
    q['A']['PTS']['books'] = ['junk', dict(book='X', line=float('nan'), over_price=-110, under_price=-110),
                              dict(book='Y', line='abc', over_price=-110, under_price=-110),
                              *q['A']['PTS']['books']]
    odds_store.save('nba', q)
    assert odds_store.load('nba')['A']['PTS']['bookmaker'] == 'FD'
    assert [r['book'] for r in line_snapshots.history('nba', 'A', 'PTS')] == ['FD']


def test_snapshot_failure_never_loses_quotes(monkeypatch):
    import sqlite3

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(line_snapshots, 'record', boom)
    odds_store.save('nba', quote(('FD', 20.5, -110, -110)))
    assert 'PTS' in odds_store.load('nba')['A']


def test_history_is_capped_to_latest_rows_in_ascending_order():
    with odds_budget.connection() as db:
        line_snapshots.record(db, 'nba', quote(('FD', 1.5, -110, -110)), now=1)
        for i in range(2, 2100):
            line_snapshots.record(db, 'nba', quote(('FD', i + 0.5, -110, -110)), now=i)
    rows = line_snapshots.history('nba', 'A', 'PTS')
    assert len(rows) == 2000 and rows[-1]['ts'] == 2099 and rows[0]['ts'] < rows[1]['ts']
    assert line_snapshots.open_lines('nba')[('A', 'PTS', 'FD')] == 1.5
