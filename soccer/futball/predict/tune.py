"""Hyperparameter and feature-group tuning.

Protocol, and the reason for it:

    fit    date <  2023-08-01              model fitting
    valid  2023-08-01 .. 2025-08-01        every decision is made here
    test   date >= 2025-08-01              touched ONCE, at the very end

Nothing — not a feature group, not a hyperparameter, not an iteration count — is
chosen using the test window. Tuning against a hold-out silently converts it into
a training set and the reported score becomes fiction.

Iteration count is found with `staged_predict`, which walks every boosting
iteration from a single fit, so tuning `max_iter` is effectively free.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss

from .. import config
from . import features

log = logging.getLogger(__name__)

VALID_FROM = "2023-08-01"
TEST_FROM = "2025-08-01"
CLASSES = ["A", "D", "H"]
MAX_GOALS = 24
MAX_ITER = 700
TUNE_DIR = config.MODELS / "predict"


@dataclass
class Split:
    fit: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    train_full: np.ndarray


def make_split(df: pl.DataFrame) -> Split:
    d = df["date"].to_numpy()
    return Split(
        fit=d < VALID_FROM,
        valid=(d >= VALID_FROM) & (d < TEST_FROM),
        test=d >= TEST_FROM,
        train_full=d < TEST_FROM,
    )


# --------------------------------------------------------------------------
# Poisson-derived quantities
# --------------------------------------------------------------------------

def poisson_grid(lam_h: np.ndarray, lam_a: np.ndarray, max_goals: int = MAX_GOALS) -> np.ndarray:
    ks = np.arange(max_goals + 1)
    ph = poisson.pmf(ks[None, :], lam_h[:, None])
    pa = poisson.pmf(ks[None, :], lam_a[:, None])
    grid = ph[:, :, None] * pa[:, None, :]
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def dixon_coles(grid: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray,
                rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction.

    Two independent Poissons systematically under-predict 0-0 and 1-1 and
    over-predict 1-0 and 0-1, because goals in a real match are not independent —
    a side that concedes early changes how the game is played. Dixon & Coles
    (1997) reweight the four lowest scorelines by a factor tau:

        tau(0,0) = 1 - lam_h*lam_a*rho     tau(0,1) = 1 + lam_h*rho
        tau(1,0) = 1 + lam_a*rho           tau(1,1) = 1 - rho

    A negative rho moves mass from 1-0/0-1 into 0-0/1-1, raising the draw
    probability. rho is fitted on the validation window, not assumed.
    """
    g = grid.copy()
    g[:, 0, 0] *= (1.0 - lam_h * lam_a * rho)
    g[:, 0, 1] *= (1.0 + lam_h * rho)
    g[:, 1, 0] *= (1.0 + lam_a * rho)
    g[:, 1, 1] *= (1.0 - rho)
    np.clip(g, 1e-12, None, out=g)
    return g / g.sum(axis=(1, 2), keepdims=True)


def grid_to_1x2(grid: np.ndarray) -> np.ndarray:
    """Columns ordered [A, D, H] to match sklearn's sorted class order."""
    n = grid.shape[1]
    iu = np.triu_indices(n, 1)
    il = np.tril_indices(n, -1)
    away = grid[:, iu[0], iu[1]].sum(1)
    home = grid[:, il[0], il[1]].sum(1)
    draw = np.trace(grid, axis1=1, axis2=2)
    p = np.vstack([away, draw, home]).T
    return p / p.sum(1, keepdims=True)


def grid_to_over(grid: np.ndarray, line: float = 2.5) -> np.ndarray:
    n = grid.shape[1]
    totals = np.add.outer(np.arange(n), np.arange(n))
    mask = (totals > line).astype(float)
    return (grid * mask[None, :, :]).sum(axis=(1, 2))


# --------------------------------------------------------------------------
# search spaces
# --------------------------------------------------------------------------

SPACE: dict[str, list[Any]] = {
    "learning_rate": [0.02, 0.04, 0.06, 0.09],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 60, 150, 300],
    "l2_regularization": [0.0, 0.5, 2.0, 8.0],
    "max_features": [0.6, 0.8, 1.0],
}


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    keys = list(SPACE)
    seen: set[tuple] = set()
    out: list[dict] = []
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        cfg = tuple(SPACE[k][int(rng.integers(len(SPACE[k])))] for k in keys)
        if cfg in seen:
            continue
        seen.add(cfg)
        out.append(dict(zip(keys, cfg)))
    return out


# --------------------------------------------------------------------------
# fitting helpers — each returns the best validation score AND best iteration
# --------------------------------------------------------------------------

def _reg(params: dict, cat: list[int]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=MAX_ITER, early_stopping=False,
        categorical_features=cat or None, random_state=42, **params,
    )


def _clf(params: dict, cat: list[int]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=MAX_ITER, early_stopping=False,
        categorical_features=cat or None, random_state=42, **params,
    )


def eval_goal_pair(
    X: np.ndarray, hg: np.ndarray, ag: np.ndarray, sp: Split,
    y: np.ndarray, over: np.ndarray, params: dict, cat: list[int],
    stride: int = 25, rhos: tuple[float, ...] = (0.0,),
    objective: str = "combined",
) -> dict:
    """Fit home+away goal regressors; pick the iteration count on validation.

    One pair of goal rates serves BOTH the 1X2 forecast and the over/under lines,
    so the default objective is the sum of their log-losses. Optimising 1X2 alone
    measurably degraded over-2.5 on the test window (-2.2pp accuracy), which is
    exactly the failure mode a single-output objective invites.
    """
    mh = _reg(params, cat).fit(X[sp.fit], hg[sp.fit])
    ma = _reg(params, cat).fit(X[sp.fit], ag[sp.fit])

    Xv = X[sp.valid]
    yv = y[sp.valid]
    ov = over[sp.valid]

    stage_h = list(mh.staged_predict(Xv))
    stage_a = list(ma.staged_predict(Xv))

    best: dict[str, Any] = {"score": np.inf}
    for i in range(stride - 1, len(stage_h), stride):
        lh = np.clip(stage_h[i], 0.05, 8.0)
        la = np.clip(stage_a[i], 0.05, 8.0)
        base = poisson_grid(lh, la)
        for rho in rhos:
            grid = base if rho == 0.0 else dixon_coles(base, lh, la, rho)
            p1x2 = np.clip(grid_to_1x2(grid), 1e-9, 1.0)
            p1x2 = p1x2 / p1x2.sum(1, keepdims=True)
            ll = log_loss(yv, p1x2, labels=CLASSES)
            po = np.clip(grid_to_over(grid), 1e-6, 1 - 1e-6)
            oll = float(log_loss(ov, po, labels=[0, 1]))
            score = ll + oll if objective == "combined" else ll
            if score < best["score"]:
                best = {
                    "score": float(score),
                    "logloss_1x2": float(ll),
                    "over_logloss": oll,
                    "n_iter": i + 1,
                    "rho": float(rho),
                    "home_mae": float(np.abs(lh - hg[sp.valid]).mean()),
                    "away_mae": float(np.abs(la - ag[sp.valid]).mean()),
                    "params": params,
                }
    return best


def eval_classifier(
    X: np.ndarray, y: np.ndarray, sp: Split, params: dict, cat: list[int],
    stride: int = 25,
) -> dict:
    m = _clf(params, cat).fit(X[sp.fit], y[sp.fit])
    Xv, yv = X[sp.valid], y[sp.valid]
    classes = list(m.classes_)
    best = {"score": np.inf}
    for i, p in enumerate(m.staged_predict_proba(Xv)):
        if (i + 1) % stride:
            continue
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum(1, keepdims=True)
        ll = log_loss(yv, p, labels=classes)
        if ll < best["score"]:
            best = {"score": float(ll), "n_iter": i + 1, "params": params}
    return best


def load() -> tuple[pl.DataFrame, Split, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = features.build()
    sp = make_split(df)
    y = df["result"].to_numpy()
    hg = df["home_goals"].to_numpy().astype(np.float64)
    ag = df["away_goals"].to_numpy().astype(np.float64)
    over = (hg + ag > 2.5).astype(int)
    return df, sp, y, hg, ag, over


def matrix(df: pl.DataFrame, groups: list[str]) -> tuple[np.ndarray, list[str], list[int]]:
    Xdf, cols = features.feature_matrix(df, groups)
    return Xdf.to_numpy().astype(np.float64), cols, features.categorical_indices(cols)


def save(name: str, payload: dict) -> None:
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    (TUNE_DIR / name).write_text(json.dumps(payload, indent=2, default=str))
