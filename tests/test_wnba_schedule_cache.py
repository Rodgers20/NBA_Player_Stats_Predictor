"""Tests for the get_todays_wnba_games TTL cache.

The WNBA player page was noticeably slower than the NBA one because
get_todays_wnba_games() issued an uncached nba_api scoreboardv3 request on
every call (~0.58s), and dashboard/app.py calls it from six separate places
per render, plus once more inside get_matchup_for_date(). The NBA side has no
equivalent per-render network call, which is why only WNBA felt slow.
"""

from datetime import date

import pytest

from utils import wnba_data_fetch


@pytest.fixture(autouse=True)
def _clear_cache():
    wnba_data_fetch.clear_schedule_cache()
    yield
    wnba_data_fetch.clear_schedule_cache()


def _patch_endpoint(monkeypatch, payload, counter):
    """Stub scoreboardv3 so no network call is made."""
    class _FakeScoreboard:
        def __init__(self, *a, **kw):
            counter.append(1)

        def get_data_frames(self):
            return payload

    import nba_api.stats.endpoints as eps
    monkeypatch.setattr(eps.scoreboardv3, "ScoreboardV3", _FakeScoreboard,
                        raising=False)
    return counter


_EMPTY = []          # <3 frames -> successful fetch, zero games


def test_second_call_hits_cache_not_network(monkeypatch):
    calls = _patch_endpoint(monkeypatch, _EMPTY, [])
    wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    assert len(calls) == 1, "repeat calls must be served from cache"


def test_distinct_dates_are_cached_separately(monkeypatch):
    calls = _patch_endpoint(monkeypatch, _EMPTY, [])
    wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    wnba_data_fetch.get_todays_wnba_games("2026-08-04")
    assert len(calls) == 2, "different dates are different cache keys"
    wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    assert len(calls) == 2, "revisiting a cached date must not refetch"


def test_expired_entry_refetches(monkeypatch):
    calls = _patch_endpoint(monkeypatch, _EMPTY, [])
    wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    assert len(calls) == 1
    # Age the entry past its TTL.
    monkeypatch.setattr(wnba_data_fetch, "_SCHEDULE_TTL", -1)
    wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    assert len(calls) == 2, "an expired entry must refetch"


def test_force_refresh_bypasses_cache(monkeypatch):
    calls = _patch_endpoint(monkeypatch, _EMPTY, [])
    wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    wnba_data_fetch.get_todays_wnba_games("2026-08-03", force_refresh=True)
    assert len(calls) == 2


def test_none_date_defaults_to_today_and_caches(monkeypatch):
    calls = _patch_endpoint(monkeypatch, _EMPTY, [])
    a = wnba_data_fetch.get_todays_wnba_games()
    b = wnba_data_fetch.get_todays_wnba_games(date.today().strftime("%Y-%m-%d"))
    assert len(calls) == 1, "None and today's explicit date share one cache key"
    assert a == b


def test_failure_is_not_cached(monkeypatch):
    """A failed fetch must not poison the cache with an empty result."""
    calls = []

    class _Boom:
        def __init__(self, *a, **kw):
            calls.append(1)
            raise RuntimeError("network down")

    import nba_api.stats.endpoints as eps
    monkeypatch.setattr(eps.scoreboardv3, "ScoreboardV3", _Boom, raising=False)

    assert wnba_data_fetch.get_todays_wnba_games("2026-08-03") == []
    assert wnba_data_fetch.get_todays_wnba_games("2026-08-03") == []
    assert len(calls) == 2, "a failure must be retried, not cached"


def test_cache_returns_equal_value(monkeypatch):
    _patch_endpoint(monkeypatch, _EMPTY, [])
    first = wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    second = wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    assert first == second
