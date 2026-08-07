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


def test_returns_empty_on_network_failure(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(wnba_injuries.requests, "get", _boom)

    assert wnba_injuries.get_wnba_team_injuries("LVA") == []
    assert wnba_injuries.get_wnba_player_injury("A'ja Wilson") is None
