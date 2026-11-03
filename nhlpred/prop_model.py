"""Player prop models: one count model per market (goals, assists, points, shots on goal).

Each market predicts the expected count mu for a player-game. Probabilities at each line come from a Poisson or
negative-binomial tail with that mu, then an isotonic map fitted on a held-out window (the last 45 game days before
the training cut-off) corrects any systematic bias. Two model families compete in the walk-forward backtest (LightGBM
with a Poisson objective and a Poisson GLM); the production model uses whichever scored better per market.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nhlpred import config as cfg
from nhlpred import evaluate as ev
from nhlpred.oddsmath import line_to_k, p_at_least
from nhlpred.playerfeat import FILL, feature_columns
from nhlpred.store import read_json, write_json

log = logging.getLogger(__name__)

MARKETS: dict[str, dict[str, Any]] = {
    "goal": {"target": "goals", "lines": [0.5], "main": 0.5, "label": "Anytime goal"},
    "sog": {"target": "sog", "lines": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], "main": 2.5, "label": "Shots on goal"},
    "assists": {"target": "assists", "lines": [0.5, 1.5], "main": 0.5, "label": "Assists"},
    "points": {"target": "points", "lines": [0.5, 1.5, 2.5], "main": 0.5, "label": "Points"},
}
CAL_GAME_DAYS = 45            # the held-out calibration window is the last 45 distinct game dates before the cut
FOLD_MONTHS = 2
TEST_FROM_SEASON = 2026       # last full season + the running one
KINDS = ("lgb", "glm")
MIN_ISO_ROWS = 2000


def _lgb() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="poisson", n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=4)


def _glm() -> Any:
    return make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=400))


def make_model(kind: str) -> Any:
    return _lgb() if kind == "lgb" else _glm()


def prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_columns()].astype(float).copy()
    for c, v in FILL.items():
        X[c] = X[c].fillna(v)
    return X.fillna(0.0)


def dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments negative-binomial alpha: var = mu + alpha mu^2; 0 means plain Poisson."""
    return float(max(0.0, (((y - mu) ** 2 - mu).sum()) / (mu**2).sum()))


def probs(mu: np.ndarray, alpha: float, lines: list[float]) -> dict[float, np.ndarray]:
    return {ln: p_at_least(mu, line_to_k(ln), alpha) for ln in lines}


def fit_iso(p: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    if len(p) < MIN_ISO_ROWS or y.sum() < 50:
        return None
    return IsotonicRegression(y_min=0.002, y_max=0.998, out_of_bounds="clip").fit(p, y)


def cal_window_start(done: pd.DataFrame, cut: pd.Timestamp) -> pd.Timestamp:
    days = np.sort(done.date[done.date < cut].unique())
    return pd.Timestamp(days[-CAL_GAME_DAYS]) if len(days) >= CAL_GAME_DAYS else pd.Timestamp(days[0])


def _fold_edges(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    test = df[(df.season >= TEST_FROM_SEASON) & df.goals.notna()]
    start = test.date.min().replace(day=1)
    edges, cur, end = [], start, test.date.max()
    while cur <= end:
        nxt = (cur + pd.DateOffset(months=FOLD_MONTHS)).replace(day=1)
        edges.append((cur, nxt))
        cur = nxt
    return edges


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample probabilities for every skater-game from ``TEST_FROM_SEASON``, long format."""
    done = df[df.goals.notna()]
    out = []
    for start, end in _fold_edges(df):
        cal_start = cal_window_start(done, start)
        train, cal, test = done[done.date < cal_start], done[(done.date >= cal_start) & (done.date < start)], done[(done.date >= start) & (done.date < end)]
        if len(test) == 0 or len(train) < 20000:
            continue
        log.info("fold %s..%s: train %s cal %s test %s", start.date(), end.date(), len(train), len(cal), len(test))
        Xtr, Xca, Xte = prep(train), prep(cal), prep(test)
        for market, m in MARKETS.items():
            tgt = m["target"]
            for kind in KINDS:
                mdl = make_model(kind).fit(Xtr, train[tgt])
                mu_c, mu_t = mdl.predict(Xca), mdl.predict(Xte)
                a = dispersion(cal[tgt].values, mu_c)
                pc, pt = probs(mu_c, a, m["lines"]), probs(mu_t, a, m["lines"])
                for ln in m["lines"]:
                    yc, yt = (cal[tgt].values >= line_to_k(ln)).astype(float), (test[tgt].values >= line_to_k(ln)).astype(float)
                    iso = fit_iso(pc[ln], yc)
                    out.append(pd.DataFrame({
                        "game_id": test.game_id.values, "player_id": test.player_id.values, "date": test.date.values, "market": market,
                        "kind": kind, "line": ln, "y": yt, "p_raw": pt[ln], "p_cal": iso.predict(pt[ln]) if iso else pt[ln],
                        "mu": mu_t, "base_mu": test[f"{tgt}_sea"].values, "alpha": a}))
    res = pd.concat(out, ignore_index=True)
    # naive baseline: the player's own (shrunk) season average through the same count tail, same dispersion
    res["p_base"] = np.nan
    for (market, ln), idx in res.groupby(["market", "line"]).groups.items():
        sub = res.loc[idx]
        res.loc[idx, "p_base"] = p_at_least(np.maximum(sub.base_mu.values, 1e-3), line_to_k(ln), float(sub.alpha.iloc[0]))
    return res


def backtest_summary(res: pd.DataFrame) -> dict[str, Any]:
    """Per market: per-line metrics for each model kind (raw and calibrated) vs. the season-average baseline."""
    out: dict[str, Any] = {}
    for market, m in MARKETS.items():
        sub = res[res.market == market]
        kinds = {}
        for kind in KINDS:
            k = sub[(sub.kind == kind) & (sub.line == m["main"])]
            kinds[kind] = {"raw": ev.summary(k.y, k.p_raw), "calibrated": ev.summary(k.y, k.p_cal)}
        best = min(KINDS, key=lambda kd: kinds[kd]["calibrated"]["log_loss"])
        use_cal = kinds[best]["calibrated"]["log_loss"] <= kinds[best]["raw"]["log_loss"]
        col = "p_cal" if use_cal else "p_raw"
        main = sub[(sub.kind == best) & (sub.line == m["main"])]
        lines = []
        for ln in m["lines"]:
            d = sub[(sub.kind == best) & (sub.line == ln)]
            lines.append({"line": ln, "n": int(len(d)), "hit_rate": round(float(d.y.mean()), 4), "avg_pred": round(float(d[col].mean()), 4),
                          **ev.summary(d.y, d[col]), "baseline": ev.summary(d.y, d.p_base)})
        out[market] = {"n": int(len(main)), "main_line": m["main"], "chosen_model": best, "calibrated": bool(use_cal),
                       "model": ev.summary(main.y, main[col]),
                       "baselines": {"season_avg": ev.summary(main.y, main.p_base),
                                     "base_rate": ev.summary(main.y, np.full(len(main), main.y.mean()))},
                       "by_kind": kinds, "calibration": ev.calibration(main.y, main[col]),
                       "top_decile": ev.top_decile(main.y, main[col]), "lines": lines}
    return out


def version(now: datetime | None = None) -> str:
    return "nhl-props-" + (now or datetime.now(UTC)).strftime("%Y%m%d")


def train(df: pd.DataFrame, summary: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Fit the production models. Train on everything before the last CAL_GAME_DAYS game days, calibrate on those days."""
    done = df[df.goals.notna()]
    cal_start = cal_window_start(done, done.date.max() + pd.Timedelta(days=1))
    train_df, cal = done[done.date < cal_start], done[done.date >= cal_start]
    Xtr, Xca = prep(train_df), prep(cal)
    now = now or datetime.now(UTC)
    bundle: dict[str, Any] = {"version": version(now), "trained_at": now.isoformat(), "features": feature_columns(), "markets": {},
                              "train_rows": int(len(train_df)), "cal_rows": int(len(cal))}
    for market, m in MARKETS.items():
        info = summary.get(market, {})
        kind, use_cal = info.get("chosen_model", "lgb"), info.get("calibrated", True)
        mdl = make_model(kind).fit(Xtr, train_df[m["target"]])
        mu_c = mdl.predict(Xca)
        a = dispersion(cal[m["target"]].values, mu_c)
        pc = probs(mu_c, a, m["lines"])
        isos = {ln: (fit_iso(pc[ln], (cal[m["target"]].values >= line_to_k(ln)).astype(float)) if use_cal else None) for ln in m["lines"]}
        bundle["markets"][market] = {"kind": kind, "model": mdl, "alpha": a, "iso": isos}
    cfg.model_dir().mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cfg.model_dir() / "props_model.joblib")
    write_json({"version": bundle["version"], "trained_at": bundle["trained_at"], "train_rows": bundle["train_rows"],
                "cal_rows": bundle["cal_rows"], "markets": {k: {"kind": v["kind"], "alpha": round(v["alpha"], 4)} for k, v in bundle["markets"].items()},
                "backtest": summary}, cfg.model_dir() / "props_summary.json", compact=False)
    return bundle


def load_bundle() -> dict[str, Any]:
    return joblib.load(cfg.model_dir() / "props_model.joblib")


def summary_json() -> dict[str, Any]:
    return read_json(cfg.model_dir() / "props_summary.json", {})


def predict(bundle: dict[str, Any], df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """market -> {'mu': array, 'p': {line: calibrated P(over)}} for the rows of ``df``."""
    X = prep(df)
    res: dict[str, dict[str, Any]] = {}
    for market, m in MARKETS.items():
        b = bundle["markets"][market]
        mu = np.asarray(b["model"].predict(X))
        raw = probs(mu, b["alpha"], m["lines"])
        res[market] = {"mu": mu, "p": {ln: (b["iso"][ln].predict(raw[ln]) if b["iso"].get(ln) is not None else raw[ln]) for ln in m["lines"]}}
    return res


def confidence(gp_l40: float) -> str:
    return "high" if gp_l40 >= 30 else "medium" if gp_l40 >= 15 else "low"
