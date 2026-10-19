"""Journal ROI breakdowns. The fixture and expected numbers mirror
frontend/lib/local-journal.test.mjs so the Python and JS summaries stay in sync."""
from utils import personal_bets as bets

ROWS = [('PTS', 'Over', -110, 'win', '2026-10-01', 'A'), ('PTS', 'Under', 120, 'loss', '2026-10-02', 'A'),
        ('REB', 'Over', -200, 'loss', '2026-10-03', 'B'), ('REB', 'Over', 250, 'push', '2026-10-04', 'B'),
        ('AST', 'Over', 150, 'win', '2026-10-05', 'A')]


def seed():
    for i, (stat, side, price, result, day, book) in enumerate(ROWS):
        entry = dict(mode='paper', league='nba', player='P', game_date=day, stat=stat, side=side,
                     line=10.5, price=price, book=book, stake=10, notes='')
        bets.settle_bet(bets.add_bet(entry, f't{i}'), result)
    return bets.summary('paper')


def test_breakdowns_by_stat_side_book_and_odds():
    s = seed()
    assert [r['key'] for r in s['by_stat']] == ['PTS', 'REB', 'AST']
    pts = s['by_stat'][0]
    assert (pts['bets'], pts['wins'], pts['losses'], pts['pushes'], pts['profit']) == (2, 1, 1, 0, -0.91)
    assert round(pts['roi'], 2) == -4.55
    assert {r['key']: (r['bets'], r['profit']) for r in s['by_side']} == {'Over': (4, 14.09), 'Under': (1, -10.0)}
    assert {r['key']: r['bets'] for r in s['by_book']} == {'A': 3, 'B': 2}
    assert [(r['key'], r['bets']) for r in s['by_odds']] == [
        ('< -150', 1), ('-150 to +100', 1), ('+100 to +200', 2), ('> +200', 1)]


def test_curve_and_max_drawdown():
    s = seed()
    assert [p['pnl'] for p in s['curve']] == [9.09, -0.91, -10.91, -10.91, 4.09]
    assert s['max_drawdown'] == 20.0


def test_odds_range_boundaries():
    assert [bets.odds_range(p) for p in (-151, -150, 100, 101, 200, 201)] == [
        '< -150', '-150 to +100', '-150 to +100', '+100 to +200', '+100 to +200', '> +200']


def test_empty_summary_has_empty_breakdowns():
    s = bets.summary('paper')
    assert s['by_stat'] == [] and s['curve'] == [] and s['max_drawdown'] == 0
