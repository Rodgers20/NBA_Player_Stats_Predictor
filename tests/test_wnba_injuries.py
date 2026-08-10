"""Tests for utils.wnba_injuries (no network)."""

import pytest

from utils import wnba_injuries


def _fake_response(json_data):
    class _R:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return json_data
    return _R()


@pytest.fixture(autouse=True)
def _reset_cache():
    wnba_injuries._cache_by_team = {}
    wnba_injuries._cache_by_player = {}
    wnba_injuries._cache_ts = None
    yield
    wnba_injuries._cache_by_team = {}
    wnba_injuries._cache_by_player = {}
    wnba_injuries._cache_ts = None


def test_parses_team_and_player_injuries(monkeypatch):
    fake = {
        "injuries": [
            {
                "displayName": "Chicago Sky",
                "abbreviation": None,
                "injuries": [
                    {
                        "athlete": {"displayName": "Skylar Diggins", "position": {"abbreviation": "G"}},
                        "status": "Out",
                        "shortComment": "Diggins (knee) out for Monday's game.",
                    },
                    {
                        "athlete": {"displayName": "Rickea Jackson", "position": {"abbreviation": "F"}},
                        "status": "Out",
                        "shortComment": "Season-ending ACL.",
                    },
                ],
            },
            {
                "displayName": "Phoenix Mercury",
                "abbreviation": None,
                "injuries": [
                    {
                        "athlete": {"displayName": "Sami Whitcomb", "position": {"abbreviation": "G"}},
                        "status": "Day-To-Day",
                        "shortComment": "Day-to-day.",
                    },
                ],
            },
        ]
    }
    monkeypatch.setattr(wnba_injuries.requests, "get", lambda *a, **kw: _fake_response(fake))

    chi = wnba_injuries.get_wnba_team_injuries("CHI")
    assert len(chi) == 2
    assert {p["name"] for p in chi} == {"Skylar Diggins", "Rickea Jackson"}
    assert all(p["status"] == "OUT" for p in chi)

    phx = wnba_injuries.get_wnba_team_injuries("PHX")
    assert len(phx) == 1
    assert phx[0]["status"] == "QUESTIONABLE"

    diggins = wnba_injuries.get_wnba_player_injury("Skylar Diggins")
    assert diggins is not None
    assert diggins["team_abbr"] == "CHI"
    assert diggins["position"] == "G"


def test_returns_empty_on_network_failure(monkeypatch, tmp_path):
    def _boom(*a, **kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(wnba_injuries.requests, "get", _boom)
    # Also point disk cache at empty tmp so no fallback data exists
    monkeypatch.setattr(wnba_injuries, "_CACHE_FILE", tmp_path / "empty.json")

    assert wnba_injuries.get_wnba_team_injuries("LVA") == []
    assert wnba_injuries.get_wnba_player_injury("A'ja Wilson") is None


def test_fuzzy_name_match_handles_hyphenated_lastname(monkeypatch):
    """Skylar Diggins ↔ Skylar Diggins-Smith should resolve to the same player."""
    fake = {
        "injuries": [
            {
                "displayName": "Chicago Sky", "abbreviation": None,
                "injuries": [
                    {"athlete": {"displayName": "Skylar Diggins", "position": {"abbreviation": "G"}},
                     "status": "Out", "shortComment": "knee"},
                ],
            }
        ]
    }
    monkeypatch.setattr(wnba_injuries.requests, "get", lambda *a, **kw: _fake_response(fake))

    assert wnba_injuries.get_wnba_player_injury("Skylar Diggins") is not None
    assert wnba_injuries.get_wnba_player_injury("Skylar Diggins-Smith") is not None  # fuzzy hit
    assert wnba_injuries.get_wnba_player_injury("Diggins") is not None                # last-only


def test_fuzzy_lookup_returns_none_for_unknown(monkeypatch):
    fake = {
        "injuries": [
            {"displayName": "Chicago Sky", "abbreviation": None,
             "injuries": [{"athlete": {"displayName": "Skylar Diggins"},
                          "status": "Out", "shortComment": "knee"}]},
        ]
    }
    monkeypatch.setattr(wnba_injuries.requests, "get", lambda *a, **kw: _fake_response(fake))
    assert wnba_injuries.get_wnba_player_injury("Random Player") is None


def test_is_player_unavailable(monkeypatch):
    fake = {
        "injuries": [
            {"displayName": "Chicago Sky", "abbreviation": None,
             "injuries": [
                 {"athlete": {"displayName": "Out Player"}, "status": "Out", "shortComment": "x"},
                 {"athlete": {"displayName": "Doubt Player"}, "status": "Doubtful", "shortComment": "x"},
                 {"athlete": {"displayName": "Day Player"}, "status": "Day-To-Day", "shortComment": "x"},
             ]},
        ]
    }
    monkeypatch.setattr(wnba_injuries.requests, "get", lambda *a, **kw: _fake_response(fake))
    assert wnba_injuries.is_player_unavailable("Out Player") is True
    assert wnba_injuries.is_player_unavailable("Doubt Player") is True
    # Day-to-Day → QUESTIONABLE → still available
    assert wnba_injuries.is_player_unavailable("Day Player") is False
    assert wnba_injuries.is_player_unavailable("Unknown") is False


def test_disk_cache_fallback(monkeypatch, tmp_path):
    """When live fetch fails but disk cache exists, use it."""
    import json
    from datetime import datetime
    cache_file = tmp_path / "injuries.json"
    payload = {"injuries": [{"displayName": "Chicago Sky", "abbreviation": None,
                             "injuries": [{"athlete": {"displayName": "Cached Player"},
                                           "status": "Out", "shortComment": "old news"}]}]}
    cache_file.write_text(json.dumps({
        "cached_at": datetime.now().isoformat(),
        "payload": payload,
    }))
    monkeypatch.setattr(wnba_injuries, "_CACHE_FILE", cache_file)

    def _fail(*a, **kw):
        raise RuntimeError("live source down")
    monkeypatch.setattr(wnba_injuries.requests, "get", _fail)

    inj = wnba_injuries.get_wnba_player_injury("Cached Player")
    assert inj is not None
    assert inj["status"] == "OUT"
