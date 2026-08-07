"""Tests for utils.wnba_odds_fetcher (no network)."""

import pytest

from utils import wnba_odds_fetcher


class _FakeResp:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status
        self.headers = {"x-requests-remaining": "999"}

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    wnba_odds_fetcher._cache = {}
    wnba_odds_fetcher._cache_ts = 0.0
    monkeypatch.setattr(wnba_odds_fetcher, "API_KEY", "test-key")
    yield
    wnba_odds_fetcher._cache = {}
    wnba_odds_fetcher._cache_ts = 0.0


def _tonight_iso():
    """A UTC ISO timestamp guaranteed to fall inside `_filter_tonight_events`'s window."""
    from datetime import datetime, timedelta, timezone
    et = datetime.now(timezone.utc) + timedelta(hours=-4)
    # Anchor tip at 8pm ET tonight (or last night's 8pm if we're in the 12am-6am gap)
    if et.hour < 6:
        et -= timedelta(days=1)
    tip_et = et.replace(hour=20, minute=0, second=0, microsecond=0)
    tip_utc = tip_et + timedelta(hours=4)
    return tip_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_returns_last_cache_without_api_key(monkeypatch):
    monkeypatch.setattr(wnba_odds_fetcher, "API_KEY", "")
    # cache is empty by default → returns empty dict
    assert wnba_odds_fetcher.get_live_wnba_odds() == {}


def test_parses_odds_from_events(monkeypatch):
    events = [
        {"id": "evt-1", "commence_time": _tonight_iso()},
        {"id": "evt-2", "commence_time": _tonight_iso()},
    ]

    evt1_data = {
        "bookmakers": [
            {
                "key": "fanduel", "title": "FanDuel",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "A'ja Wilson", "price": -115, "point": 24.5},
                            {"name": "Under", "description": "A'ja Wilson", "price": -105, "point": 24.5},
                            {"name": "Over", "description": "Caitlin Clark", "price": -110, "point": 18.5},
                            {"name": "Under", "description": "Caitlin Clark", "price": -110, "point": 18.5},
                        ],
                    },
                    {
                        "key": "player_rebounds",
                        "outcomes": [
                            {"name": "Over", "description": "A'ja Wilson", "price": 100, "point": 9.5},
                            {"name": "Under", "description": "A'ja Wilson", "price": -120, "point": 9.5},
                        ],
                    },
                ],
            }
        ]
    }
    evt2_data = {"bookmakers": []}

    def _fake_get(url, params=None, timeout=None):
        if url.endswith("/events"):
            return _FakeResp(events)
        if "evt-1" in url:
            return _FakeResp(evt1_data)
        return _FakeResp(evt2_data)

    monkeypatch.setattr(wnba_odds_fetcher.requests, "get", _fake_get)

    odds = wnba_odds_fetcher.get_live_wnba_odds()
    assert set(odds) == {"A'ja Wilson", "Caitlin Clark"}
    assert odds["A'ja Wilson"]["PTS"]["line"] == 24.5
    assert odds["A'ja Wilson"]["PTS"]["over_price"] == -115
    assert odds["A'ja Wilson"]["REB"]["line"] == 9.5
    assert odds["Caitlin Clark"]["PTS"]["line"] == 18.5
    assert odds["A'ja Wilson"]["PTS"]["bookmaker"] == "FanDuel"


def test_uses_cache_within_ttl(monkeypatch):
    call_count = {"n": 0}

    events = [{"id": "evt-1", "commence_time": _tonight_iso()}]
    evt_data = {
        "bookmakers": [{
            "key": "fanduel", "title": "FanDuel",
            "markets": [{"key": "player_points", "outcomes": [
                {"name": "Over", "description": "Player X", "price": -110, "point": 15.5},
                {"name": "Under", "description": "Player X", "price": -110, "point": 15.5},
            ]}],
        }]
    }

    def _fake_get(url, params=None, timeout=None):
        call_count["n"] += 1
        return _FakeResp(events if url.endswith("/events") else evt_data)

    monkeypatch.setattr(wnba_odds_fetcher.requests, "get", _fake_get)

    wnba_odds_fetcher.get_live_wnba_odds()
    calls_after_first = call_count["n"]
    wnba_odds_fetcher.get_live_wnba_odds()  # should hit cache — no new requests
    assert call_count["n"] == calls_after_first


def test_get_wnba_player_odds_returns_none_for_unknown(monkeypatch):
    monkeypatch.setattr(wnba_odds_fetcher, "_cache", {"A'ja Wilson": {"PTS": {"line": 24.5}}})
    monkeypatch.setattr(wnba_odds_fetcher, "_cache_ts", 9999999999.0)  # future
    assert wnba_odds_fetcher.get_wnba_player_odds("A'ja Wilson", "PTS") == {"line": 24.5}
    assert wnba_odds_fetcher.get_wnba_player_odds("Unknown Player", "PTS") is None


def test_filter_tonight_events_drops_future_and_past():
    from datetime import datetime, timedelta, timezone
    now_utc = datetime.now(timezone.utc)
    events = [
        {"id": "tonight", "commence_time": _tonight_iso()},
        {"id": "tomorrow", "commence_time": (now_utc + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")},
        {"id": "yesterday", "commence_time": (now_utc - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")},
    ]
    tonight = wnba_odds_fetcher._filter_tonight_events(events)
    ids = {e["id"] for e in tonight}
    assert "tonight" in ids
    assert "tomorrow" not in ids
    assert "yesterday" not in ids


def test_preserves_last_cache_when_all_events_401(monkeypatch):
    # Prime cache with prior good data
    prior_cache = {"Prior Player": {"PTS": {"line": 20.5, "over_price": -110, "under_price": -110, "bookmaker": "FanDuel"}}}
    monkeypatch.setattr(wnba_odds_fetcher, "_cache", dict(prior_cache))
    monkeypatch.setattr(wnba_odds_fetcher, "_cache_ts", 0.0)  # force refresh

    events = [{"id": "e1", "commence_time": _tonight_iso()}]

    def _fake_get(url, params=None, timeout=None):
        if url.endswith("/events"):
            return _FakeResp(events)
        # All per-event odds calls fail (simulate quota exhaustion)
        return None

    # Bypass _get's real implementation
    monkeypatch.setattr(wnba_odds_fetcher, "_get", _fake_get)

    result = wnba_odds_fetcher.get_live_wnba_odds()
    assert result == prior_cache, "should return the last-good cache when every event errors"
