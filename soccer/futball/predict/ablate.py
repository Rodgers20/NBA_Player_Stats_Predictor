"""Feature-group ablation, measured on the validation window only."""
from __future__ import annotations

import logging
import time

from . import tune

log = logging.getLogger(__name__)

# Incremental combinations, so each row shows what the added group is worth.
COMBOS_FORM: list[tuple[str, list[str]]] = [
    ("form (current baseline)",        ["form"]),
    ("form + elo",                     ["form", "elo"]),
    ("form + elo + venue",             ["form", "elo", "venue"]),
    ("form + elo + venue + h2h",       ["form", "elo", "venue", "h2h"]),
    ("form + elo + venue + h2h + comp", ["form", "elo", "venue", "h2h", "competition"]),
    ("elo only",                       ["elo"]),
    ("elo + comp",                     ["elo", "competition"]),
    ("form + comp",                    ["form", "competition"]),
]

BASE_PARAMS = {
    "learning_rate": 0.05, "max_leaf_nodes": 31,
    "min_samples_leaf": 20, "l2_regularization": 1.0, "max_features": 1.0,
}


def run() -> list[dict]:
    df, sp, y, hg, ag, over = tune.load()
    print(f"  rows: fit={sp.fit.sum():,}  valid={sp.valid.sum():,}  test={sp.test.sum():,} (untouched)\n")
    print(f"  {'feature groups':<34}{'cols':>5}{'1X2 ll':>10}{'O2.5 ll':>10}{'iters':>7}{'sec':>7}")
    print("  " + "-" * 73)

    results = []
    for label, groups in COMBOS_FORM:
        t0 = time.time()
        X, cols, cat = tune.matrix(df, groups)
        best = tune.eval_goal_pair(X, hg, ag, sp, y, over, BASE_PARAMS, cat)
        dt = time.time() - t0
        results.append({"label": label, "groups": groups, "n_cols": len(cols), **best})
        print(f"  {label:<34}{len(cols):>5}{best['score']:>10.4f}"
              f"{best['over_logloss']:>10.4f}{best['n_iter']:>7}{dt:>7.0f}")

    best = min(results, key=lambda r: r["score"])
    baseline = results[0]
    gain = baseline["score"] - best["score"]
    print("  " + "-" * 73)
    print(f"  best: {best['label']}  (log-loss {best['score']:.4f}, "
          f"{gain:+.4f} vs baseline {baseline['score']:.4f})")
    tune.save("ablation_form.json", {"results": results, "best": best})
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run()
