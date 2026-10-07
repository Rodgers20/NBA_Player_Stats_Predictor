"""Credits follow the league that is playing; quotes survive restarts; leans are priced."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest
from utils import odds_budget, odds_planner as planner, odds_store


def test_off_season_league_hands_its_credits_to_the_active_one():
    assert planner.split_events({'nba': 10, 'wnba': 0}, 30) == {'nba': 10, 'wnba': 0}
    assert planner.split_events({'nba': 10, 'wnba': 3}, 18) == {'nba': 3, 'wnba': 3}
    assert planner.split_events({'nba': 10, 'wnba': 1}, 18) == {'nba': 5, 'wnba': 1}
    assert planner.split_events({'nba': 4, 'wnba': 0}, 6) == {'nba': 2, 'wnba': 0}
    assert planner.split_events({'nba': 0, 'wnba': 0}, 40) == {'nba': 0, 'wnba': 0}


def test_headroom_respects_fair_share_reserve_and_daily_cap(monkeypatch):
    now = datetime(2026, 10, 6, 15, tzinfo=timezone.utc)   # 26 days left in October
    odds_budget.report('ok', 497)
    assert planner.headroom('manual', now) == 18             # (497-25)/26 rounded up
    assert planner.headroom('morning', now) == 9
    odds_budget.report('ok', 30)
    assert planner.headroom('manual', now) == 1              # 5 above the reserve, spread over 26 days
    odds_budget.report('ok', 10)
    assert planner.headroom('manual', now) == 0


def test_plan_sends_everything_to_nba_when_wnba_is_idle(monkeypatch):
    odds_budget.report('ok', 497)
    events = {'nba': [{}] * 12, 'wnba': []}
    assert planner.plan('manual', events, datetime(2026, 10, 6, 15, tzinfo=timezone.utc)) == {'nba': 6, 'wnba': 0}


def test_due_kind_morning_then_pretip():
    et = ZoneInfo('America/New_York')
    early = datetime(2026, 10, 6, 7, tzinfo=et)
    tip = datetime(2026, 10, 6, 19, tzinfo=et)
    assert planner.due_kind(early, tip, set()) is None
    assert planner.due_kind(early.replace(hour=10), tip, set()) == 'morning'
    assert planner.due_kind(early.replace(hour=10), tip, {'morning'}) is None
    assert planner.due_kind(tip - timedelta(hours=1), tip, {'morning'}) == 'pretip'
    assert planner.due_kind(tip - timedelta(hours=1), tip, {'morning', 'pretip'}) is None
    assert planner.due_kind(tip + timedelta(minutes=1), tip, {'morning'}) is None


def test_quotes_survive_restart_and_newer_replace_older():
    odds_store.save('nba', {'A': {'PTS': {'line': 20.5}}})
    odds_store.save('nba', {'A': {'PTS': {'line': 21.5}, 'REB': {'line': 5.5}}, 'B': {'AST': {'line': 6.5}}})
    odds_store.save('wnba', {'C': {'PTS': {'line': 12.5}}})
    nba = odds_store.load('nba')
    assert nba['A'] == {'PTS': {'line': 21.5}, 'REB': {'line': 5.5}} and nba['B']['AST']['line'] == 6.5
    assert list(odds_store.load('wnba')) == ['C']


def test_refresh_skips_provider_when_no_events(monkeypatch):
    called = []
    monkeypatch.setattr(planner, '_module', lambda league: called.append(league))
    assert planner.refresh('nba', 0) == {}
    assert called == []


@pytest.mark.parametrize('ev,reason,kind,eligible,price_shown', [
    (0.12, None, 'pick', True, True),
    (-0.04, None, 'lean', False, True),     # priced but no edge: still shows price, probability, EV
    (0.12, 'Quote expired; refresh or enter the current book price', 'research', False, False),
])
def test_priced_rows_are_picks_or_leans_and_unpriced_rows_are_research(ev, reason, kind, eligible, price_shown):
    from api.routes.props import _serialise_prop
    row = _serialise_prop(dict(player='A', stat='PTS', direction='Over', line=20.5, ev=ev, model_prob=0.6,
                               model_projection=23.1, live_over_price=-110, has_live_odds=True), reason)
    assert row['pick_type'] == kind and row['recommendation_eligible'] is eligible
    assert (row['price'] == -110) is price_shown and (row['ev'] == ev) is price_shown
    assert row['model_projection'] == 23.1


def test_next_slate_falls_forward_to_the_first_day_with_games():
    from utils.slate import next_slate
    games = {'2026-10-07': ['NYL@ATL', 'LVA@GSV']}
    assert next_slate(lambda day: games.get(day, []), '2026-10-06') == ('2026-10-07', games['2026-10-07'])
    assert next_slate(lambda day: games.get(day, []), '2026-10-07')[0] == '2026-10-07'   # today wins when it has games
    assert next_slate(lambda day: [], '2026-10-06') == ('2026-10-06', [])                 # nothing nearby: stay on today
    assert next_slate(lambda day: games.get(day, []), '2026-10-01') == ('2026-10-01', [])  # beyond the lookahead window


def test_wnba_tip_time_is_eastern_wall_clock_not_utc():
    from utils.wnba_data_fetch import _et_iso
    assert _et_iso('2026-10-07T19:30:00Z') == '2026-10-07T19:30:00-04:00'   # 7:30 pm ET == 23:30Z
    assert _et_iso('2026-12-07T19:30:00Z') == '2026-12-07T19:30:00-05:00'   # standard time
    assert _et_iso('') == '' and _et_iso('not a time') == 'not a time'


def test_manual_refresh_buys_the_next_slate_the_day_before(monkeypatch):
    from datetime import datetime, timedelta, timezone
    odds_budget.report('ok', 497)
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=23, minute=30).isoformat()
    monkeypatch.setattr(planner, 'upcoming', lambda league, target=None: [{'commence_time': tomorrow}] * 2 if league == 'wnba' else [])
    calls = []
    monkeypatch.setattr(planner, 'refresh', lambda league, count, target=None: calls.append((league, count, target)))
    planner.manual_refresh('wnba')
    assert calls == [('wnba', 2, planner.event_date({'commence_time': tomorrow}))]
