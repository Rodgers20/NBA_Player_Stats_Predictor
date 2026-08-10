"""Compute WNBA hit rates: which "X+ stat" thresholds each player has cleared
in most of their last N games. Used by the /wnba/hitrates page.

Example output for a player:
    {
      "player_name": "Kelsey Mitchell",
      "team": "IND",
      "games": 10,
      "entries": [
        {"stat": "PTS", "threshold": 20, "hits": 10, "games": 10},
        {"stat": "FG3M", "threshold": 3, "hits": 10, "games": 10},
        {"stat": "AST", "threshold": 2, "hits": 8, "games": 10},
        ...
      ]
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class HitRateEntry:
    player_name: str
    team: str
    game_id: str          # opponent/date so the UI can group by game
    game_label: str       # e.g. "LVA @ IND"
    stat: str             # "PTS" | "REB" | "AST" | "FG3M"
    threshold: int        # X in "X+ Stat"
    hits: int
    games: int

    @property
    def ratio(self) -> float:
        return self.hits / self.games if self.games else 0.0


# Stat labels for the UI ("3+ Threes Made" reads better than "3+ FG3M")
STAT_LABELS = {
    "PTS":  "Points",
    "REB":  "Rebounds",
    "AST":  "Assists",
    "FG3M": "Threes Made",
}


def _best_thresholds_for_stat(values: pd.Series, stat: str, min_hits: int) -> list[tuple[int, int, int]]:
    """Return the most impressive "X+ Stat" hits: for each hit-count tier
    (n, n-1, ..., min_hits), the HIGHEST threshold the player cleared that
    many times.

    Filters out trivially-low thresholds: a player who averages 12 REB
    doesn't need a "5+ REB 10/10" line — that's dead money. Any surfaced
    threshold must be at least 60% of the player's average for the window.
    """
    if values.empty:
        return []
    n = len(values)
    hi = int(values.max())
    avg = float(values.mean())
    floors = {"PTS": 8, "REB": 3, "AST": 2, "FG3M": 1}
    floor = floors.get(stat, 1)
    # Enforce minimum meaningful threshold relative to player's own avg
    value_floor = max(floor, int(0.6 * avg))

    # For every hit-count h in [n .. min_hits], find the max threshold with hits>=h.
    tiers: list[tuple[int, int, int]] = []
    seen_hits: set[int] = set()
    for h in range(n, min_hits - 1, -1):
        max_t = 0
        for t in range(hi, value_floor - 1, -1):
            if int((values >= t).sum()) >= h:
                max_t = t
                break
        if max_t >= value_floor and h not in seen_hits:
            tiers.append((max_t, h, n))
            seen_hits.add(h)

    # Also dedupe threshold: if 20+ Pts 10/10 also holds at 9/10 (obviously),
    # keep only the top tier per threshold value.
    dedup: dict[int, tuple[int, int, int]] = {}
    for t, h, g in tiers:
        prev = dedup.get(t)
        if prev is None or h > prev[1]:
            dedup[t] = (t, h, g)
    return sorted(dedup.values(), key=lambda x: (-x[1], -x[0]))


def compute_hit_rates(
    wnba_df: pd.DataFrame,
    todays_games: list[dict],
    n_games: int = 10,
    min_hits: int = 8,
    min_avg_min: float = 15.0,
) -> list[dict]:
    """Compute per-game hit-rate blocks for tonight's WNBA games.

    Returns a list of game dicts:
        [
          {
            "matchup": "LVA @ IND",
            "home": "IND", "away": "LVA",
            "entries": [ HitRateEntry, ... ]   # sorted by ratio DESC then threshold DESC
          },
          ...
        ]

    Only players with L20 avg minutes >= `min_avg_min` are included.
    Only "X+ Stat" thresholds hit in >= `min_hits`/n_games are surfaced.
    """
    if wnba_df.empty or not todays_games:
        return []

    df_sorted = wnba_df.sort_values(["PLAYER_NAME", "_date"], ascending=[True, False])
    recent_by_player: dict[str, pd.DataFrame] = {
        name: group.head(n_games) for name, group in df_sorted.groupby("PLAYER_NAME")
    }
    l20_by_player: dict[str, pd.DataFrame] = {
        name: group.head(20) for name, group in df_sorted.groupby("PLAYER_NAME")
    }

    results: list[dict] = []
    for g in todays_games:
        home = g["home"]["abbrev"]
        away = g["away"]["abbrev"]
        matchup = f"{away} @ {home}"
        game_id = f"{away}-{home}"

        entries: list[HitRateEntry] = []
        for player_name, recent in recent_by_player.items():
            if len(recent) < n_games:
                continue
            team = recent.iloc[0].get("TEAM_ABBREVIATION", "")
            if team not in (home, away):
                continue
            l20 = l20_by_player.get(player_name)
            if l20 is None or l20["MIN"].mean() < min_avg_min:
                continue

            for stat in ("PTS", "REB", "AST", "FG3M"):
                if stat not in recent.columns:
                    continue
                values = recent[stat].dropna()
                if values.empty:
                    continue
                for threshold, hits, games in _best_thresholds_for_stat(values, stat, min_hits):
                    entries.append(HitRateEntry(
                        player_name=player_name,
                        team=team,
                        game_id=game_id,
                        game_label=matchup,
                        stat=stat,
                        threshold=threshold,
                        hits=hits,
                        games=games,
                    ))

        # Sort: highest ratio first, then highest threshold, then player name
        entries.sort(key=lambda e: (-e.ratio, -e.threshold, e.player_name))

        if entries:
            results.append({
                "matchup": matchup,
                "home": home,
                "away": away,
                "entries": entries,
            })

    return results
