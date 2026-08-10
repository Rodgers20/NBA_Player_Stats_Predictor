"""Tests for utils.league_config."""

import pytest
from utils.league_config import get_config, all_leagues, NBA, WNBA


def test_all_leagues_returns_both():
    leagues = all_leagues()
    assert set(leagues) == {"nba", "wnba"}


def test_get_config_nba():
    cfg = get_config("nba")
    assert cfg.key == "nba"
    assert cfg.display_name == "NBA"
    assert cfg.data_source == "kaggle"
    assert cfg.api_league_id == "00"
    assert cfg.season_start_month == 10
    assert len(cfg.team_abbrevs) == 30  # NBA has 30 teams


def test_get_config_wnba():
    cfg = get_config("wnba")
    assert cfg.key == "wnba"
    assert cfg.display_name == "WNBA"
    assert cfg.data_source == "nba_api"
    assert cfg.api_league_id == "10"
    assert cfg.season_start_month == 5
    assert 12 <= len(cfg.team_abbrevs) <= 16  # WNBA: 13 current + PDX/TOR expansion


def test_get_config_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown league"):
        get_config("nhl")  # type: ignore[arg-type]


def test_team_ids_in_expected_range():
    for cfg in (NBA, WNBA):
        for abbrev, tid in cfg.team_abbrevs.items():
            assert cfg.team_id_min <= tid <= cfg.team_id_max, (
                f"{cfg.key} team {abbrev}={tid} outside range "
                f"[{cfg.team_id_min}, {cfg.team_id_max}]"
            )


def test_data_dirs_are_league_scoped():
    nba_dir = get_config("nba").data_dir
    wnba_dir = get_config("wnba").data_dir
    assert nba_dir != wnba_dir
    assert nba_dir.name == "nba"
    assert wnba_dir.name == "wnba"


def test_configs_are_immutable():
    """LeagueConfig is frozen — mutation should fail."""
    cfg = get_config("nba")
    with pytest.raises(Exception):  # dataclass FrozenInstanceError
        cfg.key = "wnba"  # type: ignore[misc]
