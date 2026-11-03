"""Tune both goal-model families on validation only.

The historical test window has been consulted in earlier development. This
search never reads its outcomes. Run train afterwards to save diagnostic
held-out predictions and refit the production models.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from .. import config
from . import features, tune

log = logging.getLogger(__name__)

MODEL_DIR = config.MODELS / "predict"
RHOS = (0.0, -0.02, -0.04, -0.06, -0.08, -0.10, -0.13, -0.16)
N_CONFIGS = 24

FORM_GROUPS_DEFAULT = ["form", "elo", "venue", "h2h", "competition"]


def _best_groups(fallback: list[str]) -> list[str]:
    """Pick the feature set by COMBINED 1X2 + over-2.5 log-loss.

    The goal models serve both outputs, so optimising one in isolation picks the
    wrong set: 'form + elo' edges 1X2 by 0.00002 but loses 0.001 on over-2.5.
    """
    path = tune.TUNE_DIR / "ablation_form.json"
    if path.exists():
        try:
            rows = json.loads(path.read_text())["results"]
            best = min(rows, key=lambda r: r["score"] + r["over_logloss"])
            return best["groups"]
        except Exception:  # noqa: BLE001
            pass
    return fallback


def search_goals(df, sp, y, hg, ag, over, groups: list[str], label: str) -> dict:
    X, cols, cat = tune.matrix(df, groups)
    configs = tune.sample_configs(N_CONFIGS, seed=7)
    print(f"\n  [{label}] {len(cols)} features, {len(configs)} configs")
    best: dict[str, Any] = {"score": np.inf}
    t0 = time.time()
    for i, params in enumerate(configs, 1):
        r = tune.eval_goal_pair(X, hg, ag, sp, y, over, params, cat, rhos=RHOS,
                                objective="combined")
        if r["score"] < best["score"]:
            best = {**r, "groups": groups, "columns": cols}
            print(f"    {i:>2}/{len(configs)}  combined={r['score']:.4f} "
                  f"(1x2={r['logloss_1x2']:.4f} o25={r['over_logloss']:.4f})"
                  f"  iters={r['n_iter']}  rho={r['rho']:+.2f}   <- best")
    print(f"    searched in {time.time()-t0:.0f}s")
    return best


def search_clf(df, sp, y, groups: list[str], label: str) -> dict:
    X, cols, cat = tune.matrix(df, groups)
    configs = tune.sample_configs(N_CONFIGS, seed=11)
    print(f"\n  [{label}] {len(cols)} features, {len(configs)} configs")
    best: dict[str, Any] = {"score": np.inf}
    t0 = time.time()
    for i, params in enumerate(configs, 1):
        r = tune.eval_classifier(X, y, sp, params, cat)
        if r["score"] < best["score"]:
            best = {**r, "groups": groups, "columns": cols}
            print(f"    {i:>2}/{len(configs)}  ll={r['score']:.4f}  iters={r['n_iter']}   <- best")
    print(f"    searched in {time.time()-t0:.0f}s")
    return best


def _scaled_iters(n_iter: int, sp: tune.Split) -> int:
    """More training rows warrant slightly more boosting rounds."""
    ratio = sp.train_full.sum() / max(sp.fit.sum(), 1)
    return int(round(n_iter * min(ratio, 1.5)))


def run() -> dict:
    df, sp, y, hg, ag, over = tune.load()
    print("=" * 74)
    print("  MODEL TUNING")
    print("=" * 74)
    print(f"  fit={sp.fit.sum():,}  valid={sp.valid.sum():,}  test={sp.test.sum():,} (untouched)")

    form_groups = _best_groups(FORM_GROUPS_DEFAULT)
    market_groups = sorted(set(form_groups) | {"market"})
    print(f"  form groups   : {form_groups}")
    print(f"  market groups : {market_groups}")

    best_form = search_goals(df, sp, y, hg, ag, over, form_groups, "form-only goal models")
    best_mkt = search_goals(df, sp, y, hg, ag, over, market_groups, "market-aware goal models")

    out = {
        "form": best_form, "market": best_mkt,
        "form_groups": form_groups, "market_groups": market_groups,
        "forecast": "coherent goal grid",
    }
    tune.save("tuning.json", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run()
