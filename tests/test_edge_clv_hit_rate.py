import pytest
from utils import clv
from utils.edge_rating import edge_fields, edge_stars
from utils.hit_rate_windows import h2h_hit_rate, hit_rate_windows


@pytest.mark.parametrize('pct,stars', [(-0.1, None), (None, None), (0, 1), (0.9, 1), (1, 2),
                                       (2.5, 3), (3, 4), (4.9, 4), (5, 5), (12, 5)])
def test_stars_thresholds(pct, stars):
    assert edge_stars(pct) == stars


def test_edge_fields():
    out = edge_fields(0.034, 0.52)
    assert out == dict(edge_pct=3.4, implied_prob=0.52, stars=4, high_edge=True)
    assert edge_fields(0.01, 0.5)['high_edge'] is False
    assert edge_fields(None, None) == dict(edge_pct=None, implied_prob=None, stars=None, high_edge=False)


def test_hit_windows_push_is_neither_hit_nor_miss():
    values = [25, 20, 18, 30, 21] + [10] * 20   # most recent first
    out = hit_rate_windows(values, 20, 'Over')
    assert out['L5'] == dict(hits=3, total=4, pct=75.0)      # the 20 is a push
    assert out['L10']['total'] == 9 and out['L20']['total'] == 19
    assert out['season']['total'] == 24
    under = hit_rate_windows(values, 20, 'Under')
    assert under['L5'] == dict(hits=1, total=4, pct=25.0)


def test_hit_windows_short_and_empty():
    assert hit_rate_windows([], 10, 'Over')['L5'] == dict(hits=0, total=0, pct=None)
    out = hit_rate_windows([12, 8], 10, 'Over', season_values_desc=[12])
    assert out['L5']['total'] == 2 and out['season']['total'] == 1


def test_h2h():
    assert h2h_hit_rate([22, 18, 25], 20, 'over') == dict(hits=2, total=3, pct=66.7)


def test_clv_same_line_uses_price_gain():
    out = clv.compute_clv('Over', 20.5, -110, 20.5, -130)
    assert out['close_line'] == 20.5 and out['close_price'] == -130
    assert out['clv_pct'] == pytest.approx(100 * (130 / 230 - 110 / 210), abs=0.01)
    assert out['clv_pct'] > 0


def test_clv_line_move_direction():
    over = clv.compute_clv('Over', 20.5, -110, 22.5, -110)['clv_pct']
    under = clv.compute_clv('Under', 20.5, -110, 22.5, -110)['clv_pct']
    assert over == pytest.approx(4 * clv.LINE_HALF_POINT_PCT) and under == pytest.approx(-over)


def test_clv_missing_close_is_none_and_bad_side_raises():
    assert clv.compute_clv('Over', 20.5, -110, None, None) == dict(clv.EMPTY)
    with pytest.raises(ValueError):
        clv.compute_clv('Sideways', 20.5, -110, 20.5, -110)


SNAPS = [dict(book='A', line=20.5, over_price=-110, under_price=-110, ts=100),
         dict(book='A', line=21.5, over_price=-120, under_price=100, ts=200),
         dict(book='B', line=21.5, over_price=-105, under_price=-115, ts=200),
         dict(book='A', line=23.5, over_price=-110, under_price=-110, ts=900)]


def test_closing_quote_respects_cutoff_and_book():
    assert clv.closing_quote(SNAPS, 50) is None
    assert clv.closing_quote(SNAPS, 500, 'A')['line'] == 21.5
    assert clv.closing_quote(SNAPS, 500)['line'] == 21.5            # consensus across books
    out = clv.clv_from_snapshots('Under', 20.5, -110, SNAPS, 500, 'A')
    assert out['close_line'] == 21.5 and out['close_price'] == 100
