"""Pre-game team features. Every value for a game uses only games that came before it (shift(1) before any window).

Priors are fixed league-wide constants, not fitted on the data, so no future information leaks through them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRIORS = {"gf": 3.05, "ga": 3.05, "sf": 29.5, "sa": 29.5, "win": 0.5}
SHRINK_K = 6
WINDOWS = (10, 30)
SV_PRIOR, SV_PRIOR_SHOTS = 0.905, 300.0
MAX_REST = 7


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game (played or not), sorted by team and time."""
    done = games.home_score.notna()
    cols = []
    for side, other in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "game_id": games.game_id, "date": pd.to_datetime(games.date), "start_utc": games.start_utc,
            "season": games.season, "team_id": games[f"{side}_id"], "opp_id": games[f"{other}_id"], "home": side == "home",
            "gf": games[f"{side}_score"], "ga": games[f"{other}_score"],
            "sf": games.get(f"{side}_sog"), "sa": games.get(f"{other}_sog")})
        d["win"] = np.where(done, (d.gf > d.ga).astype(float), np.nan)
        cols.append(d)
    t = pd.concat(cols, ignore_index=True).sort_values(["team_id", "start_utc", "game_id"]).reset_index(drop=True)
    gap = t.groupby("team_id").date.diff().dt.days
    t["rest"] = gap.fillna(MAX_REST).clip(upper=MAX_REST)
    t["b2b"] = (t.rest <= 1).astype(int)
    return t


def add_rolling(t: pd.DataFrame) -> pd.DataFrame:
    """Shrunk rolling means of gf/ga/sf/sa/win over the last 10 and 30 games, strictly pre-game."""
    out = t.copy()
    grp = out.groupby("team_id", sort=False)
    for s, prior in PRIORS.items():
        x = grp[s].shift(1)
        seen = x.notna().astype(float)
        for w in WINDOWS:
            tot = x.groupby(out.team_id).transform(lambda v: v.rolling(w, min_periods=1).sum())
            cnt = seen.groupby(out.team_id).transform(lambda v: v.rolling(w, min_periods=1).sum())
            out[f"{s}{w}"] = (tot.fillna(0) + SHRINK_K * prior) / (cnt + SHRINK_K)
    return out


def starters(goalies: pd.DataFrame) -> pd.DataFrame:
    """The goalie who started each team-game (flagged starter, else most ice time)."""
    if goalies.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "goalie_id"])
    g = goalies.sort_values(["game_id", "team_id", "starter", "toi"], ascending=[True, True, False, False])
    g = g.drop_duplicates(["game_id", "team_id"])
    return g[["game_id", "team_id", "player_id"]].rename(columns={"player_id": "goalie_id"})


def goalie_pre_svp(goalies: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Save% over each goalie's previous ``window`` appearances, shrunk to the league rate; strictly pre-game."""
    g = goalies.sort_values(["player_id", "date", "game_id"]).copy()
    for c in ("saves", "shots_against"):
        x = g.groupby("player_id")[c].shift(1)
        g[c + "_r"] = x.groupby(g.player_id).transform(lambda v: v.rolling(window, min_periods=1).sum()).fillna(0)
    g["svp_pre"] = (g.saves_r + SV_PRIOR * SV_PRIOR_SHOTS) / (g.shots_against_r + SV_PRIOR_SHOTS)
    return g[["game_id", "player_id", "svp_pre"]]


def goalie_latest_svp(goalies: pd.DataFrame, window: int = 10) -> dict[int, float]:
    """Each goalie's save% over their most recent ``window`` appearances (for games not played yet)."""
    g = goalies.sort_values(["player_id", "date", "game_id"]).groupby("player_id").tail(window)
    s = g.groupby("player_id")[["saves", "shots_against"]].sum()
    return ((s.saves + SV_PRIOR * SV_PRIOR_SHOTS) / (s.shots_against + SV_PRIOR_SHOTS)).to_dict()


def projected_starters(goalies: pd.DataFrame, panel: pd.DataFrame) -> dict[int, int]:
    """Team -> goalie most likely to start next: the one with most starts in the last 10 team games.

    On a back-to-back, if that goalie also started the last game, the next-busiest goalie is projected instead.
    """
    out: dict[int, int] = {}
    st = starters(goalies).merge(goalies[["game_id", "date"]].drop_duplicates("game_id"), on="game_id")
    for team, g in st.sort_values(["date", "game_id"]).groupby("team_id"):
        recent = g.tail(10)
        counts = recent.goalie_id.value_counts()
        pick = int(counts.index[0])
        out[int(team)] = pick
        last = int(g.goalie_id.iloc[-1])
        nxt = panel[(panel.team_id == team) & panel.gf.isna()].head(1)
        if len(counts) > 1 and last == pick and len(nxt) and int(nxt.b2b.iloc[0]) == 1:
            out[int(team)] = int(counts.index[1])
    return out


def team_panel(games: pd.DataFrame, goalies: pd.DataFrame) -> pd.DataFrame:
    """Team-game rows with rolling features and the starting goalie's pre-game save%."""
    t = add_rolling(team_games(games))
    if goalies.empty:
        t["goalie_id"], t["svp"] = np.nan, SV_PRIOR
        return t
    st = starters(goalies).merge(goalie_pre_svp(goalies), left_on=["game_id", "goalie_id"], right_on=["game_id", "player_id"], how="left")
    t = t.merge(st[["game_id", "team_id", "goalie_id", "svp_pre"]], on=["game_id", "team_id"], how="left")
    latest, proj = goalie_latest_svp(goalies), projected_starters(goalies, t)
    future = t.gf.isna()
    t.loc[future, "goalie_id"] = t.loc[future, "team_id"].map(proj)
    t["svp"] = np.where(future, t.goalie_id.map(latest), t.svp_pre)
    t["svp"] = t.svp.fillna(SV_PRIOR)
    t["goalie_projected"] = future
    return t.drop(columns="svp_pre")
