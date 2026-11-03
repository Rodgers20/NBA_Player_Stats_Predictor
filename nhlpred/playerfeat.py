"""Pre-game player features (rolling rates shrunk to position priors) for the prop models.

One code path builds both the training rows (games already played) and the rows to predict (projected lineups of
upcoming games, stats = NaN), so the two can never disagree about what a feature means. Every window is computed on
values shifted by one game, so a row never sees its own game or anything after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nhlpred.teamfeat import SV_PRIOR

STATS = ["goals", "assists", "points", "sog", "toi", "pp_goals"]
WINDOWS = (5, 10, 20)
K_WIN, K_SEASON, K_CAREER = 5.0, 10.0, 25.0
POS_CODE = {"C": 0, "L": 1, "R": 2, "D": 3}
TEAM_CTX = ["gf10", "gf30", "sf10", "sf30", "rest", "b2b"]
OPP_CTX = ["ga10", "ga30", "sa10", "sa30", "svp", "rest", "b2b"]
LINEUP_LAST, LINEUP_MIN_OF_LAST = 5, 3


def _cum_window(x0: pd.Series, seen: pd.Series, keys: pd.Series, w: int) -> tuple[pd.Series, pd.Series]:
    """Sum and count of the previous ``w`` rows per player, from cumulative sums (fast, no per-group python)."""
    cs, cn = x0.groupby(keys).cumsum(), seen.groupby(keys).cumsum()
    return cs - cs.groupby(keys).shift(w).fillna(0), cn - cn.groupby(keys).shift(w).fillna(0)


def position_priors(p: pd.DataFrame) -> pd.DataFrame:
    """Mean of each stat by position and date, using strictly earlier dates only."""
    done = p[p.goals.notna()]
    daily = done.groupby(["pos_code", "date"])[STATS].sum().join(done.groupby(["pos_code", "date"]).size().rename("n"))
    cum = daily.groupby(level=0).cumsum().reset_index()
    for s in STATS:
        cum[f"pri_{s}"] = cum[s] / cum.n
    cum = cum[["pos_code", "date"] + [f"pri_{s}" for s in STATS]].sort_values("date")
    return cum


def projected_lineups(skaters: pd.DataFrame, upcoming: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Skater rows (stats NaN) for each upcoming game: players who dressed in >=3 of the team's last 5 games and in
    at least one of its last two. The free API has no scratch/injury list, so this is a projection."""
    rows = []
    last_games = team_games[team_games.gf.notna()].sort_values(["start_utc", "game_id"]).groupby("team_id").tail(LINEUP_LAST)
    for team_id, g in last_games.groupby("team_id"):
        ids = g.game_id.tolist()
        sub = skaters[skaters.game_id.isin(ids) & (skaters.team_id == team_id)]
        counts = sub.groupby("player_id").game_id.nunique()
        recent2 = set(sub[sub.game_id.isin(ids[-2:])].player_id)
        ok = [pid for pid, c in counts.items() if c >= LINEUP_MIN_OF_LAST and pid in recent2]
        latest = sub.sort_values(["date", "game_id"]).drop_duplicates("player_id", keep="last").set_index("player_id")
        for _, ug in upcoming[(upcoming.home_id == team_id) | (upcoming.away_id == team_id)].iterrows():
            home = ug.home_id == team_id
            for pid in ok:
                r = latest.loc[pid]
                rows.append({"game_id": ug.game_id, "date": ug.date, "start_utc": ug.start_utc, "season": ug.season,
                             "player_id": pid, "name": r["name"], "pos": r.pos, "team": r.team, "team_id": team_id,
                             "opp": ug.away_abbr if home else ug.home_abbr, "opp_id": ug.away_id if home else ug.home_id,
                             "home": bool(home)})
    return pd.DataFrame(rows)


def history_rows(skaters: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "start_utc", "home_id", "away_id"]]
    h = skaters.merge(g, on="game_id")
    h["opp_id"] = np.where(h.home, h.away_id, h.home_id)
    return h.drop(columns=["home_id", "away_id"])


def build(skaters: pd.DataFrame, games: pd.DataFrame, panel: pd.DataFrame, future: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for every skater-game in ``skaters`` plus any ``future`` (projected) rows. ``panel`` = team_panel()."""
    hist = history_rows(skaters, games)
    p = pd.concat([hist, future], ignore_index=True) if future is not None and len(future) else hist
    p["date"] = pd.to_datetime(p.date)
    p["pos_code"] = p.pos.map(POS_CODE).fillna(0).astype(int)
    p = p.sort_values(["player_id", "start_utc", "game_id"]).reset_index(drop=True)
    pri = position_priors(p)
    p = pd.merge_asof(p.sort_values("date"), pri, on="date", by="pos_code", allow_exact_matches=False)
    p = p.sort_values(["player_id", "start_utc", "game_id"]).reset_index(drop=True)
    for c in STATS:                                  # first-ever dates have no prior yet: fall back to a sane constant
        p[f"pri_{c}"] = p[f"pri_{c}"].fillna({"goals": .12, "assists": .2, "points": .32, "sog": 1.8, "toi": 15, "pp_goals": .03}[c])
    keys = p.player_id
    feats: dict[str, pd.Series] = {}
    sk = pd.Series(p.season.astype(str) + "_" + p.player_id.astype(str))
    for s in STATS:
        x0 = p.groupby("player_id")[s].shift(1)
        has = x0.notna().astype(float)
        x0 = x0.fillna(0)
        pr = p[f"pri_{s}"]
        for w in WINDOWS:
            tot, cnt = _cum_window(x0, has, keys, w)
            feats[f"{s}_l{w}"] = (tot + K_WIN * pr) / (cnt + K_WIN)
        tot_s, cnt_s = x0.groupby(sk).cumsum(), has.groupby(sk).cumsum()
        feats[f"{s}_sea"] = (tot_s + K_SEASON * pr) / (cnt_s + K_SEASON)
        tot_c, cnt_c = x0.groupby(keys).cumsum(), has.groupby(keys).cumsum()
        feats[f"{s}_car"] = (tot_c + K_CAREER * pr) / (cnt_c + K_CAREER)
        if s == "toi":
            t3, c3 = _cum_window(x0, has, keys, 3)
            feats["toi_l3"] = (t3 + 3 * pr) / (c3 + 3)
            feats["gp_car"] = cnt_c
            feats["gp_l20"] = _cum_window(x0, has, keys, 20)[1]
            feats["gp_l40"] = _cum_window(x0, has, keys, 40)[1]
        if s == "sog":                               # shots per 60 minutes over the last 20 games, shrunk 3 pseudo-games
            sog20, _ = _cum_window(x0, has, keys, 20)
            toi20, _ = _cum_window(p.groupby("player_id").toi.shift(1).fillna(0), has, keys, 20)
            feats["sog60_l20"] = (sog20 + 3 * 2.0) / ((toi20 + 3 * 15.0) / 60)
    p = pd.concat([p, pd.DataFrame(feats)], axis=1)
    # shooting percentage over the last 40 games (shots-weighted, shrunk to 10%)
    ones = pd.Series(1.0, index=p.index)
    g40, _ = _cum_window(p.groupby("player_id").goals.shift(1).fillna(0), ones, keys, 40)
    s40, _ = _cum_window(p.groupby("player_id").sog.shift(1).fillna(0), ones, keys, 40)
    p["sh_pct40"] = (g40 + 0.10 * 40) / (s40 + 40)
    p["pp_share20"] = (p.pp_goals_l20 + 0.0) / (p.goals_l20 + 0.05)
    # context from the team panel
    own = panel[["game_id", "team_id"] + TEAM_CTX].rename(columns={c: f"own_{c}" for c in TEAM_CTX})
    opp = panel[["game_id", "team_id"] + OPP_CTX].rename(columns={c: f"opp_{c}" for c in OPP_CTX})
    p = p.merge(own, on=["game_id", "team_id"], how="left").merge(opp, left_on=["game_id", "opp_id"], right_on=["game_id", "team_id"],
                                                                 how="left", suffixes=("", "_o")).drop(columns="team_id_o")
    p["home_i"] = p.home.astype(int)
    return p.sort_values(["start_utc", "game_id", "player_id"]).reset_index(drop=True)


def feature_columns() -> list[str]:
    cols = [f"{s}_{k}" for s in STATS for k in ("l5", "l10", "l20", "sea", "car")]
    cols += ["toi_l3", "gp_car", "gp_l20", "gp_l40", "sog60_l20", "sh_pct40", "pp_share20", "pos_code", "home_i"]
    cols += [f"own_{c}" for c in TEAM_CTX] + [f"opp_{c}" for c in OPP_CTX]
    return cols


FILL = {"opp_svp": SV_PRIOR}
