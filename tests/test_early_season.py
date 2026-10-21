import pandas as pd
import pytest

from utils.early_season import shrink_early_season, current_season_games

BASELINE = 2.5


def forecast(margin=-9.0):
    return dict(margin=margin, margin_sd=12.0, home_win_probability=0.2,
                prediction_intervals={'margin_80': [margin - 15, margin + 15], 'total_80': [-20, 20]})


def history(counts):
    rows = []
    for team, n in counts.items():
        rows += [dict(team=team, date=pd.Timestamp('2026-10-01') + pd.Timedelta(days=i),
                      season=2026, playoffs=0) for i in range(n)]
        rows += [dict(team=team, date=pd.Timestamp('2026-04-01'), season=2025, playoffs=0)] * 82
        rows += [dict(team=team, date=pd.Timestamp('2026-05-20'), season=2025, playoffs=1)] * 5
    return pd.DataFrame(rows)


def test_counts_only_current_regular_season_games():
    h = history({'WAS': 3, 'MIL': 10})
    assert current_season_games(h, 'WAS', pd.Timestamp('2026-10-21')) == 3
    assert current_season_games(h, 'MIL', pd.Timestamp('2026-10-21')) == 10


def test_wnba_season_is_calendar_year_and_unknown_history_is_none():
    h = pd.DataFrame([dict(team='NYL', date=pd.Timestamp('2026-05-20') + pd.Timedelta(days=i), playoffs=0)
                      for i in range(4)] + [dict(team='NYL', date=pd.Timestamp('2025-09-01'), playoffs=0)])
    assert current_season_games(h, 'NYL', pd.Timestamp('2026-06-10'), 'wnba') == 4
    assert current_season_games(pd.DataFrame({'PTS': [1]}), 'NYL', pd.Timestamp('2026-06-10'), 'wnba') is None


def test_opening_week_pulls_hard_toward_baseline():
    out = shrink_early_season(forecast(), 3, BASELINE)
    assert out['margin'] == pytest.approx(BASELINE + (-9 - BASELINE) * 3 / 15)
    assert 0.4 < out['home_win_probability'] < 0.6
    # interval stays centred on the shrunk margin with the same width
    lo, hi = out['prediction_intervals']['margin_80']
    assert (lo + hi) / 2 == pytest.approx(out['margin'])
    assert hi - lo == pytest.approx(30)


def test_full_trust_late_in_season_and_input_not_mutated():
    original = forecast()
    out = shrink_early_season(original, 70, BASELINE)
    assert out['margin'] == pytest.approx(BASELINE + (-9 - BASELINE) * 70 / 82)
    assert original['margin'] == -9.0 and original['home_win_probability'] == 0.2


def test_zero_games_is_pure_baseline():
    assert shrink_early_season(forecast(), 0, BASELINE)['margin'] == pytest.approx(BASELINE)
