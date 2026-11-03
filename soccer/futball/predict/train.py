"""Fit the production models from the tuned configuration and report honestly.

The hyperparameters, feature groups, boosting iterations and Dixon-Coles rho
come from `tune_all.py`, which chose them using a validation
window that ends before the test window begins. This module therefore reads the
historical test window for diagnostics; earlier development already consulted it.

Models produced:

  home/away_goals        Poisson goal rates, market features included
  home/away_goals_form   Poisson goal rates, no market features
  home/away_corners      Poisson corner rates, form features only
  home/away_sot          Poisson shots-on-target rates, form features only

Corners and shots on target are fitted on the 27,210 domestic matches that
carry them; the Champions League publishes neither, so those markets are absent
there rather than extrapolated from a different competition. Odds are not used:
bookmakers price goals, and the closing 1X2 line says little about corner counts
that the rolling corner averages do not already say.

The served forecast is derived from the goal rates via a Dixon-Coles-corrected
score grid, so win probabilities, expected goals, over/under lines and the
likeliest scoreline are all consequences of the same two numbers and cannot
contradict each other.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from .. import config
from ..storage import atomic_path, write_text_atomic
from . import features, tune

log = logging.getLogger(__name__)

MODEL_DIR = config.MODELS / "predict"
CLASSES = tune.CLASSES
TEST_FROM = tune.TEST_FROM
VALID_FROM = tune.VALID_FROM

DEFAULTS: dict[str, Any] = {
    "params": {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20,
               "l2_regularization": 1.0, "max_features": 1.0},
    "n_iter": 400,
    "rho": 0.0,
}


@dataclass
class Metrics:
    n_train: int
    n_test: int
    test_from: str
    feature_groups: list[str]
    n_features: int
    dixon_coles_rho: float
    dixon_coles_rho_market: float
    result_accuracy: float
    result_logloss: float
    market_accuracy: float
    market_logloss: float
    blend_weight: float
    blend_accuracy: float
    blend_logloss: float
    formonly_accuracy: float
    formonly_logloss: float
    home_goals_mae: float
    away_goals_mae: float
    total_goals_mae: float
    over25_accuracy: float
    over25_logloss: float
    over25_form_accuracy: float
    over25_form_logloss: float
    corners_total_mae: float | None
    corners_n_train: int
    sot_total_mae: float | None
    sot_n_train: int
    secondary_competitions: list[str]
    verdict: str
    methodology_note: str


def _tuning() -> dict:
    path = MODEL_DIR / "tuning.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            log.warning("tuning.json unreadable; falling back to defaults")
    return {}


def _cfg(t: dict, key: str) -> tuple[dict, int, float]:
    node = t.get(key) or {}
    return (
        node.get("params") or DEFAULTS["params"],
        int(node.get("n_iter") or DEFAULTS["n_iter"]),
        float(node.get("rho") or 0.0),
    )


#: The per-team count markets, and the lake columns that carry them.
SECONDARY_MARKETS = {
    "corners": ("home_corners", "away_corners"),
    "sot": ("home_sot", "away_sot"),
}


def _fit_secondary(df, X, cat, params, n_iter, tr, te) -> dict:
    """Poisson rate models for the per-team count markets.

    Each is fitted only on rows where **both** sides' counts are recorded, so a
    half-published match cannot teach the model that a team had zero corners
    when the truth is that nobody wrote the number down.
    """
    out: dict = {"competitions": []}
    covered: set[str] = set()
    codes = df["competition_code"].to_numpy()

    for market, (home_col, away_col) in SECONDARY_MARKETS.items():
        h = df[home_col].to_numpy().astype(np.float64)
        a = df[away_col].to_numpy().astype(np.float64)
        have = np.isfinite(h) & np.isfinite(a)
        fit_rows, test_rows = tr & have, te & have
        if int(fit_rows.sum()) < 500:
            log.warning("%s: only %d usable rows; market not fitted", market, int(fit_rows.sum()))
            out[market] = {"home": None, "away": None, "total_mae": None,
                       "n_train": int(fit_rows.sum()), "held_out": {}}
            continue

        def reg(n):
            m = tune._reg(params, cat)
            m.set_params(max_iter=n)
            return m

        eh = reg(n_iter).fit(X[fit_rows], h[fit_rows])
        ea = reg(n_iter).fit(X[fit_rows], a[fit_rows])
        # Predictions on the held-out window, from the model that never saw it.
        # These are frozen alongside the goal ones and are what the calibration
        # loop later measures its bias against — an in-sample comparison would
        # report a model with no drift because it had already fitted the drift.
        mae = None
        held: dict[int, tuple[float, float]] = {}
        if test_rows.any():
            ph = np.clip(eh.predict(X[test_rows]), 0.05, 30.0)
            pa = np.clip(ea.predict(X[test_rows]), 0.05, 30.0)
            mae = round(float(mean_absolute_error(h[test_rows] + a[test_rows], ph + pa)), 4)
            held = {int(i): (float(x), float(y))
                    for i, x, y in zip(np.flatnonzero(test_rows), ph, pa)}

        # Refit on every usable row for serving, the way the goal models do.
        all_rows = have
        out[market] = {
            "home": reg(n_iter).fit(X[all_rows], h[all_rows]),
            "away": reg(n_iter).fit(X[all_rows], a[all_rows]),
            "total_mae": mae,
            "n_train": int(fit_rows.sum()),
            "held_out": held,
        }
        covered |= set(codes[all_rows].tolist())

    out["competitions"] = sorted(covered)
    return out


def train() -> Metrics:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t = _tuning()

    form_groups = t.get("form_groups") or ["form", "elo", "venue", "h2h", "competition"]
    market_groups = t.get("market_groups") or sorted(set(form_groups) | {"market"})

    df = features.build(include_style="style" in form_groups or "style" in market_groups)
    sp = tune.make_split(df)
    y = df["result"].to_numpy()
    hg = df["home_goals"].to_numpy().astype(np.float64)
    ag = df["away_goals"].to_numpy().astype(np.float64)
    over_true = ((hg + ag) > 2.5).astype(int)

    Xf, cols_f, cat_f = tune.matrix(df, form_groups)
    Xm, cols_m, cat_m = tune.matrix(df, market_groups)

    # The two goal-model families tuned their own rho on validation (-0.10 for
    # form-only, -0.13 with market features). Using one for both would apply a
    # correction fitted for a different model.
    p_form, it_form, rho = _cfg(t, "form")
    p_mkt, it_mkt, rho_mkt = _cfg(t, "market")

    scale = 1.0  # Keep the iteration count selected on validation.
    it_form_f = int(round(it_form * scale))
    it_mkt_f = int(round(it_mkt * scale))

    def reg(params, cat, n):
        m = tune._reg(params, cat)
        m.set_params(max_iter=n)
        return m

    tr, te = sp.train_full, sp.test

    # ---------------- fit on train_full, then read the test window ONCE -------
    fh = reg(p_form, cat_f, it_form_f).fit(Xf[tr], hg[tr])
    fa = reg(p_form, cat_f, it_form_f).fit(Xf[tr], ag[tr])
    mh = reg(p_mkt, cat_m, it_mkt_f).fit(Xm[tr], hg[tr])
    ma = reg(p_mkt, cat_m, it_mkt_f).fit(Xm[tr], ag[tr])

    lam_h = np.clip(fh.predict(Xf[te]), 0.05, 8.0)
    lam_a = np.clip(fa.predict(Xf[te]), 0.05, 8.0)
    grid = tune.poisson_grid(lam_h, lam_a)
    if rho:
        grid = tune.dixon_coles(grid, lam_h, lam_a, rho)
    p_form_1x2 = np.clip(tune.grid_to_1x2(grid), 1e-9, 1.0)
    p_form_1x2 = p_form_1x2 / p_form_1x2.sum(1, keepdims=True)

    form_acc = float(accuracy_score(y[te], np.array(CLASSES)[p_form_1x2.argmax(1)]))
    form_ll = float(log_loss(y[te], p_form_1x2, labels=CLASSES))

    # Over/under is served from whichever goal-model family matches the
    # information available, so BOTH are measured. Reporting only one and
    # comparing it against the other family's historical number would look like
    # a regression that never happened.
    p_over_form = np.clip(tune.grid_to_over(grid), 1e-6, 1 - 1e-6)
    of_acc = float(accuracy_score(over_true[te], (p_over_form > 0.5).astype(int)))
    of_ll = float(log_loss(over_true[te], p_over_form, labels=[0, 1]))

    mlam_h = np.clip(mh.predict(Xm[te]), 0.05, 8.0)
    mlam_a = np.clip(ma.predict(Xm[te]), 0.05, 8.0)
    mgrid = tune.poisson_grid(mlam_h, mlam_a)
    if rho_mkt:
        mgrid = tune.dixon_coles(mgrid, mlam_h, mlam_a, rho_mkt)
    p_over = np.clip(tune.grid_to_over(mgrid), 1e-6, 1 - 1e-6)
    o_acc = float(accuracy_score(over_true[te], (p_over > 0.5).astype(int)))
    o_ll = float(log_loss(over_true[te], p_over, labels=[0, 1]))
    hg_mae = float(mean_absolute_error(hg[te], mlam_h))
    ag_mae = float(mean_absolute_error(ag[te], mlam_a))
    tot_mae = float(mean_absolute_error(hg[te] + ag[te], mlam_h + mlam_a))

    proba = tune.grid_to_1x2(mgrid)
    res_acc = float(accuracy_score(y[te], np.array(CLASSES)[proba.argmax(1)]))
    res_ll = float(log_loss(y[te], proba, labels=CLASSES))

    # ---------------- corners and shots on target -----------------------------
    # Separate Poisson rates rather than anything derived from the goal grid:
    # a match can be corner-heavy and goalless. Fitted only on rows that carry
    # the target, which excludes the Champions League entirely.
    secondary = _fit_secondary(df, Xf, cat_f, p_form, it_form_f, tr, te)

    mkt = df.select(["mkt_away", "mkt_draw", "mkt_home"]).to_numpy()[te]
    ok = ~np.isnan(mkt).any(axis=1)
    mkt_n = mkt[ok] / mkt[ok].sum(1, keepdims=True)
    mkt_acc = float(accuracy_score(y[te][ok], np.array(CLASSES)[mkt_n.argmax(1)]))
    mkt_ll = float(log_loss(y[te][ok], mkt_n, labels=CLASSES))

    # The served odds-informed forecast is the goal grid, not a separate
    # classifier blend. Compare the market on identical odds-available rows.
    w = 1.0
    blend = np.where(ok[:, None], proba, p_form_1x2)
    blend_acc = float(accuracy_score(y[te], np.array(CLASSES)[blend.argmax(1)]))
    blend_ll = float(log_loss(y[te], blend, labels=CLASSES))

    metrics = Metrics(
        n_train=int(tr.sum()), n_test=int(te.sum()), test_from=TEST_FROM,
        feature_groups=form_groups, n_features=len(cols_f), dixon_coles_rho=rho,
        dixon_coles_rho_market=rho_mkt,
        result_accuracy=round(res_acc, 4), result_logloss=round(res_ll, 4),
        market_accuracy=round(mkt_acc, 4), market_logloss=round(mkt_ll, 4),
        blend_weight=round(w, 3), blend_accuracy=round(blend_acc, 4),
        blend_logloss=round(blend_ll, 4),
        formonly_accuracy=round(form_acc, 4), formonly_logloss=round(form_ll, 4),
        home_goals_mae=round(hg_mae, 4), away_goals_mae=round(ag_mae, 4),
        total_goals_mae=round(tot_mae, 4),
        over25_accuracy=round(o_acc, 4), over25_logloss=round(o_ll, 4),
        over25_form_accuracy=round(of_acc, 4), over25_form_logloss=round(of_ll, 4),
        corners_total_mae=secondary["corners"]["total_mae"],
        corners_n_train=secondary["corners"]["n_train"],
        sot_total_mae=secondary["sot"]["total_mae"],
        sot_n_train=secondary["sot"]["n_train"],
        secondary_competitions=secondary["competitions"],
        verdict="Probabilities come from one score distribution. Historical evaluation is diagnostic; it does not establish a betting edge.",
        methodology_note=(
            "Configurations were selected on validation dates "
            f"{VALID_FROM} to {TEST_FROM}. Evaluation models train only before "
            f"{TEST_FROM}; their predictions are saved before the all-data production refit. "
            "This historical test window was consulted during earlier development and "
            "again during this repair, so these are diagnostic results, not an untouched test. "
            "Closing-odds evaluation represents information available close to kick-off, "
            "not necessarily days before a fixture."
        ),
    )

    # Freeze evaluation predictions BEFORE fitting production models on all rows.
    # These files are never generated from the production bundle.
    test_rows = df.filter(pl.Series(te)).to_dicts()
    test_index = np.flatnonzero(te)
    frozen = []
    for i, r in enumerate(test_rows):
        row_at = int(test_index[i])
        counts = {}
        for market, (home_col, away_col) in SECONDARY_MARKETS.items():
            pair = secondary[market].get("held_out", {}).get(row_at)
            if pair is None:
                continue
            actual_h, actual_a = r.get(home_col), r.get(away_col)
            if actual_h is None or actual_a is None:
                continue
            counts[market] = {
                "predicted_home": round(pair[0], 3), "predicted_away": round(pair[1], 3),
                "actual_home": int(actual_h), "actual_away": int(actual_a),
            }
        pp = blend[i]
        actual = {"H": "home", "D": "draw", "A": "away"}[r["result"]]
        picked = ["away", "draw", "home"][int(pp.argmax())]
        frozen.append({
            "competition_code": r["competition_code"], "date": r["date"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "actual_home": r["home_goals"], "actual_away": r["away_goals"],
            "actual_result": actual, "predicted": picked, "correct": picked == actual,
            "win": dict(zip(["away", "draw", "home"], pp.tolist())),
            "expected_total": float((mlam_h + mlam_a)[i] if ok[i] else (lam_h + lam_a)[i]),
            "actual_total": r["total_goals"],
            "over_2_5": float(p_over[i] if ok[i] else p_over_form[i]),
            "actual_over_2_5": bool(over_true[te][i]),
            "counts": counts,
            "trained_before": TEST_FROM,
        })
    write_text_atomic(MODEL_DIR / "backtest.json", json.dumps({"trained_before": TEST_FROM, "rows": frozen}))
    empirical = np.array([(y[tr] == c).mean() for c in CLASSES])
    baseline = np.tile(empirical, (int(te.sum()), 1))
    onehot = np.column_stack([y[te] == c for c in CLASSES]).astype(float)
    bins = []
    confidence, correct = blend.max(1), blend.argmax(1) == onehot.argmax(1)
    for lo in np.arange(0.3, 1.0, 0.1):
        mask = (confidence >= lo) & (confidence < lo + 0.1)
        if mask.any():
            bins.append({"from": round(float(lo), 1), "n": int(mask.sum()),
                         "confidence": float(confidence[mask].mean()), "accuracy": float(correct[mask].mean())})
    diagnostics = {
        "baseline_accuracy": float(accuracy_score(y[te], np.array(CLASSES)[baseline.argmax(1)])),
        "baseline_logloss": float(log_loss(y[te], baseline, labels=CLASSES)),
        "brier_score": float(((blend - onehot) ** 2).sum(1).mean()),
        "market_comparison_n": int(ok.sum()),
        "served_logloss_odds_subset": float(log_loss(y[te][ok], blend[ok], labels=CLASSES)),
        "calibration": bins,
        "last_training_result": str(df["date"].max()),
    }

    # ---------------- refit on everything for serving -------------------------
    bundle = {
        "trained_through": str(df["date"].max()),
        "home_goals_form": reg(p_form, cat_f, it_form_f).fit(Xf, hg),
        "away_goals_form": reg(p_form, cat_f, it_form_f).fit(Xf, ag),
        "home_goals": reg(p_mkt, cat_m, it_mkt_f).fit(Xm, hg),
        "away_goals": reg(p_mkt, cat_m, it_mkt_f).fit(Xm, ag),
        "form_columns": cols_f,
        "columns": cols_m,
        "form_groups": form_groups,
        "market_groups": market_groups,
        "classes": list(CLASSES),
        "blend_weight": w,
        "rho": rho,
        "rho_market": rho_mkt,
        "categorical_form": cat_f,
        "categorical_market": cat_m,
        "home_corners": secondary["corners"]["home"],
        "away_corners": secondary["corners"]["away"],
        "home_sot": secondary["sot"]["home"],
        "away_sot": secondary["sot"]["away"],
        # Which competitions the corner and shot models actually saw. Serving a
        # corner line for a competition absent from this list would be an
        # extrapolation from other leagues dressed up as a forecast.
        "secondary_competitions": secondary["competitions"],
    }
    with atomic_path(MODEL_DIR / "models.joblib") as temporary:
        joblib.dump(bundle, temporary)
    write_text_atomic(MODEL_DIR / "metrics.json", json.dumps({**asdict(metrics), **diagnostics}, indent=2))
    return metrics


def report() -> int:
    prev_path = MODEL_DIR / "metrics_before_tuning.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else None

    m = train()
    print("\n  PREDICTION MODEL REPORT")
    print("  " + "-" * 68)
    print(f"  features              {m.n_features} across {m.feature_groups}")
    print(f"  train rows            {m.n_train:,}")
    print(f"  test rows             {m.n_test:,}   (held out from {m.test_from})")
    print(f"  Dixon-Coles rho       {m.dixon_coles_rho:+.2f}")
    print()

    def line(label: str, acc: float, ll: float, key_a: str | None = None,
             key_l: str | None = None) -> None:
        delta = ""
        if prev and key_a and key_l and key_a in prev:
            da = acc - prev[key_a]
            dl = ll - prev[key_l]
            delta = f"   ({da:+.3f} acc, {dl:+.4f} ll vs before)"
        print(f"  {label:<22}acc {acc:.3f}   log-loss {ll:.4f}{delta}")

    line("closing-odds market", m.market_accuracy, m.market_logloss)
    line("model (with odds)", m.result_accuracy, m.result_logloss,
         "result_accuracy", "result_logloss")
    line("served forecast", m.blend_accuracy, m.blend_logloss,
         "blend_accuracy", "blend_logloss")
    line("form-only (served)", m.formonly_accuracy, m.formonly_logloss,
         "formonly_accuracy", "formonly_logloss")
    line("over 2.5 (with odds)", m.over25_accuracy, m.over25_logloss,
         "over25_accuracy", "over25_logloss")
    line("over 2.5 (form only)", m.over25_form_accuracy, m.over25_form_logloss)
    print()
    print(f"  goal MAE              home {m.home_goals_mae:.3f}  "
          f"away {m.away_goals_mae:.3f}  total {m.total_goals_mae:.3f}")
    for label, mae, n in (("corners", m.corners_total_mae, m.corners_n_train),
                          ("shots on target", m.sot_total_mae, m.sot_n_train)):
        if mae is None:
            print(f"  {label + ' MAE':<22}not fitted ({n:,} usable rows)")
        else:
            print(f"  {label + ' MAE':<22}total {mae:.3f}   ({n:,} training rows)")
    print(f"  count markets cover   {', '.join(m.secondary_competitions) or 'nothing'}")
    print()
    print(f"  verdict: {m.verdict}")
    print()
    print("  note:", m.methodology_note[:78])
    for i in range(78, len(m.methodology_note), 78):
        print("       ", m.methodology_note[i:i + 78])
    print("  " + "-" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
