"""League configuration registry.

Central source of truth for league-specific constants (data source, team IDs,
CDN URL templates, directory paths). Consumed by data loaders, feature
engineering, dashboard routing, and model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

League = Literal["nba", "wnba"]

_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class LeagueConfig:
    key: League
    display_name: str
    brand: str
    data_source: Literal["kaggle", "nba_api"]
    kaggle_dataset: str
    api_league_id: str
    season_start_month: int
    season_format: Literal["hyphenated", "single_year"]
    team_id_min: int
    team_id_max: int
    team_abbrevs: Mapping[str, int]
    logo_cdn_template: str
    headshot_cdn_template: str
    position_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {"guard_min_ast": 4.0, "center_min_reb": 7.0}
    )

    @property
    def data_dir(self) -> Path:
        return _PROJECT_ROOT / "data" / self.key

    @property
    def models_dir(self) -> Path:
        return _PROJECT_ROOT / "models" / self.key


NBA = LeagueConfig(
    key="nba",
    display_name="NBA",
    brand="NBA Props AI",
    data_source="kaggle",
    kaggle_dataset="eoinamoore/historical-nba-data-and-player-box-scores",
    api_league_id="00",
    season_start_month=10,
    season_format="hyphenated",
    team_id_min=1610612737,
    team_id_max=1610612766,
    team_abbrevs={
        "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
        "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
        "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
        "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
        "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
        "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
        "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
        "UTA": 1610612762, "WAS": 1610612764,
    },
    logo_cdn_template="https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg",
    headshot_cdn_template="https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png",
    position_thresholds={"guard_min_ast": 4.0, "center_min_reb": 7.0},
)

WNBA = LeagueConfig(
    key="wnba",
    display_name="WNBA",
    brand="WNBA Props AI",
    data_source="nba_api",
    kaggle_dataset="",
    api_league_id="10",
    season_start_month=5,
    season_format="single_year",
    team_id_min=1611661313,
    team_id_max=1611661332,
    team_abbrevs={
        "ATL": 1611661330, "CHI": 1611661329, "CON": 1611661323, "DAL": 1611661321,
        "GSV": 1611661331, "IND": 1611661325, "LAS": 1611661320, "LVA": 1611661319,
        "MIN": 1611661324, "NYL": 1611661313, "PDX": 1611661327, "PHX": 1611661317,
        "SEA": 1611661328, "TOR": 1611661332, "WAS": 1611661322,
    },
    logo_cdn_template="https://cdn.wnba.com/logos/wnba/{team_id}/global/L/logo.svg",
    headshot_cdn_template="https://cdn.wnba.com/headshots/wnba/latest/1040x760/{player_id}.png",
    position_thresholds={"guard_min_ast": 3.5, "center_min_reb": 6.5},
)


_CONFIGS: dict[League, LeagueConfig] = {"nba": NBA, "wnba": WNBA}


def get_config(league: League) -> LeagueConfig:
    if league not in _CONFIGS:
        raise ValueError(f"Unknown league: {league!r}. Expected one of {list(_CONFIGS)}.")
    return _CONFIGS[league]


def all_leagues() -> list[League]:
    return list(_CONFIGS.keys())
