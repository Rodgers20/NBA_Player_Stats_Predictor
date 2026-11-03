"""Serving layer for the prediction models.

Everything a prediction reports — win probabilities, expected goals, over/under
lines, both-teams-to-score and the likeliest scoreline — is derived from ONE
pair of Poisson goal rates through a Dixon-Coles-corrected score grid. They are
therefore consequences of the same two numbers and cannot contradict one
another, which an earlier design (separate 1X2 and goals models) did in 6% of
matches.

Serve-time features are produced by `features.build(extra_fixtures=...)`, i.e.
the identical code path that produced the training features. Every plausible
pairing in every competition is pushed through it once and cached, so a request
is a table lookup plus a model call rather than a feature-engineering
re-implementation. Reconstructing features on the serving side is the classic
source of silent train/serve skew.

Where valid pre-match odds exist they are inputs to the goal regressors.
Otherwise the form-only goal regressors are used. All outputs share one grid.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

import joblib
import numpy as np
import polars as pl

from .. import config
from ..cache import file_cache
from . import calibration, features, reasons, tune
from .train import MODEL_DIR

log = logging.getLogger(__name__)

MAX_PAIRS_PER_COMPETITION = 40  # teams; 40 -> 1,560 ordered pairings


class ModelsUnavailable(RuntimeError):
    pass


@file_cache(lambda: MODEL_DIR / "models.joblib")
def _bundle() -> dict[str, Any]:
    path = MODEL_DIR / "models.joblib"
    if not path.exists():
        raise ModelsUnavailable(
            "Prediction models are not trained yet. Run: python -m futball.predict.train"
        )
    return joblib.load(path)


@file_cache(lambda: MODEL_DIR / "metrics.json")
def _training_metadata() -> dict:
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@file_cache(lambda: config.SILVER / "results" / "results.parquet")
def _results() -> pl.DataFrame:
    return pl.read_parquet(config.SILVER / "results" / "results.parquet")


@file_cache(lambda: config.SILVER / "results" / "results.parquet")
def _played() -> pl.DataFrame:
    """Feature table for matches that were actually played (used for backtests)."""
    return features.build()


def _recent_teams(df: pl.DataFrame, code: str, limit: int) -> list[str]:
    """Clubs selectable for a forecast: this season's, then last season's.

    The current season comes first and in full. Ranking the last two seasons
    together by match count looks reasonable and is wrong for a cup: in
    September a Champions League newcomer has played once while last season's
    semi-finalist has thirteen behind them, so the cut kept the clubs who are
    *not in the competition any more* and dropped most of the ones who are. The
    Champions League board could forecast three of eighteen fixtures.
    """
    sub = df.filter(pl.col("competition_code") == code)
    if sub.height == 0:
        return []

    seasons = sorted(sub["season"].unique().to_list())[-2:]

    def clubs_in(season_list: list[str]) -> dict[str, int]:
        frame = sub.filter(pl.col("season").is_in(season_list))
        counts: dict[str, int] = {}
        for column in ("home_team", "away_team"):
            for name, n in frame[column].value_counts().iter_rows():
                if name:
                    counts[name] = counts.get(name, 0) + n
        return counts

    current = clubs_in(seasons[-1:])
    chosen = sorted(sorted(current, key=lambda n: (-current[n], n))[:limit])
    if len(chosen) >= limit or len(seasons) < 2:
        return chosen

    # Fill the rest with last season's clubs, busiest first; the name breaks
    # ties so the selection does not move with Python's hash seed.
    previous = {n: c for n, c in clubs_in(seasons[:-1]).items() if n not in set(chosen)}
    room = limit - len(chosen)
    chosen += sorted(previous, key=lambda n: (-previous[n], n))[:room]
    return sorted(chosen)


@file_cache(lambda: config.SILVER / "results" / "results.parquet", lambda: MODEL_DIR / "models.joblib", daily=True)
def _fixture_table() -> pl.DataFrame:
    """Features for every plausible pairing, built by the training pipeline."""
    res = _results()
    today = dt.date.today().isoformat()
    fixtures: list[dict] = []
    for code in sorted(res["competition_code"].unique().to_list()):
        meta = res.filter(pl.col("competition_code") == code).tail(1).to_dicts()[0]
        teams = _recent_teams(res, code, MAX_PAIRS_PER_COMPETITION)
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                fixtures.append({
                    "competition_code": code,
                    "competition": meta["competition"],
                    "country": meta["country"],
                    "season": meta["season"],
                    "date": today,
                    "home_team": h,
                    "away_team": a,
                })
    built = features.build(extra_fixtures=fixtures, include_style="style" in (_bundle().get("form_groups", []) + _bundle().get("market_groups", [])))
    return built.filter(pl.col("is_fixture"))


def warm() -> dict:
    """Force the caches to build; returns a small summary."""
    b = _bundle()
    ft = _fixture_table()
    return {"fixtures_cached": ft.height, "features": len(b["form_columns"])}


def competitions() -> list[dict]:
    df = _played()
    return (
        df.group_by(["competition_code", "competition", "country"])
        .agg(
            pl.len().alias("matches"),
            pl.col("date").max().alias("last_match"),
            pl.col("season").last().alias("latest_season"),
        )
        .sort("competition")
        .to_dicts()
    )


def teams(competition_code: str) -> list[str]:
    """Clubs selectable for a prediction — last two seasons.

    Early in a season only a handful of clubs have played, so restricting to the
    current season alone would leave most of a division unpickable.
    """
    return _recent_teams(_results(), competition_code, MAX_PAIRS_PER_COMPETITION)


@file_cache(lambda: config.SILVER / "results" / "results.parquet", lambda: MODEL_DIR / "models.joblib", daily=True, maxsize=128)
def _row_for(competition_code: str, home: str, away: str, fixture_date: str | None = None) -> pl.DataFrame:
    ft = _fixture_table()
    row = ft.filter(
        (pl.col("competition_code") == competition_code)
        & (pl.col("home_team") == home)
        & (pl.col("away_team") == away)
    )
    if row.height and (fixture_date is None or fixture_date == dt.date.today().isoformat()):
        return row.tail(1)
    # Not in the precomputed grid (rare) — build this one fixture on demand.
    res = _results().filter(pl.col("competition_code") == competition_code)
    if res.height == 0:
        raise ValueError(f"unknown competition {competition_code}")
    meta = res.tail(1).to_dicts()[0]
    built = features.build(extra_fixtures=[{
        "competition_code": competition_code, "competition": meta["competition"],
        "country": meta["country"], "season": meta["season"],
        "date": fixture_date or dt.date.today().isoformat(), "home_team": home, "away_team": away,
    }], include_style="style" in (_bundle().get("form_groups", []) + _bundle().get("market_groups", [])))
    row = built.filter(pl.col("is_fixture"))
    if row.height == 0:
        raise ValueError(f"could not build features for {home} v {away}")
    return row.tail(1)


def _vector(row: pl.DataFrame, cols: list[str]) -> np.ndarray:
    vals = []
    d = row.to_dicts()[0]
    for c in cols:
        v = d.get(c)
        vals.append(float(v) if isinstance(v, (int, float)) else np.nan)
    return np.array([vals], dtype=np.float64)


# --------------------------------------------------------------------------
# Corners and shots on target
#
# These are per-team counts, not consequences of the goal grid: a match can be
# corner-heavy and goalless. Each side gets its own Poisson rate, and the match
# total is the convolution of the two — which is just a Poisson on the summed
# rate, so no grid is needed.
# --------------------------------------------------------------------------

#: The over/under line quoted for each market's match total. Chosen as the
#: half-point nearest the long-run mean of the lake, so neither side is a
#: foregone conclusion: domestic football averages 9.7 corners and 8.6 shots on
#: target per match.
COUNT_LINES = {"corners": 9.5, "sot": 8.5}

#: "Both teams to reach N" is quoted at the largest N that both sides clear on
#: their own expectation, so the claim is one the model actually believes.
_MAX_THRESHOLD = 12


def _poisson_tail(rate: float, at_least: int) -> float:
    """P(X >= at_least) for a Poisson rate."""
    from scipy.stats import poisson
    return float(poisson.sf(at_least - 1, rate))


def _count_market(
    market: str, bundle: dict, vector: np.ndarray, code: str,
    home_team: str, away_team: str, table: dict,
) -> dict:
    """One count market, or an honest absence."""
    home_model = bundle.get(f"home_{market}")
    away_model = bundle.get(f"away_{market}")
    covered = bundle.get("secondary_competitions") or []
    noun = "Corners" if market == "corners" else "Shots on target"

    if home_model is None or away_model is None:
        return {"available": False, "note": f"{noun} were not fitted in this model build."}
    if code not in covered:
        return {
            "available": False,
            "note": (f"{config.LEAGUES.get(code, {}).get('name', code)} does not publish "
                     f"{noun.lower()}, and borrowing another competition's rate would "
                     f"be a guess."),
        }

    correction = calibration.correction_for(market, code, table)
    factor = float(correction["factor"]) if correction else 1.0

    lam_h = float(np.clip(home_model.predict(vector)[0], 0.05, 30.0)) * factor
    lam_a = float(np.clip(away_model.predict(vector)[0], 0.05, 30.0)) * factor
    line = COUNT_LINES[market]
    total_rate = lam_h + lam_a

    # The biggest whole number both sides are expected to reach.
    threshold = min(int(lam_h), int(lam_a), _MAX_THRESHOLD)
    both = None
    if threshold >= 1:
        both = {
            "threshold": threshold,
            # Independent per side: the models are fitted separately and the
            # lake gives no within-match dependence to estimate.
            "probability": round(_poisson_tail(lam_h, threshold) * _poisson_tail(lam_a, threshold), 4),
        }

    def side(team: str, rate: float) -> dict:
        target = max(1, int(round(rate)))
        return {
            "team": team,
            "expected": round(rate, 2),
            "line": target,
            "at_least": round(_poisson_tail(rate, target), 4),
            "distribution": [
                {"count": k, "probability": round(_poisson_tail(rate, k), 4)}
                for k in range(1, min(int(rate) + 5, 16))
            ],
        }

    return {
        "available": True,
        "home": side(home_team, lam_h),
        "away": side(away_team, lam_a),
        "total": {
            "expected": round(total_rate, 2),
            "line": line,
            "over": round(_poisson_tail(total_rate, int(line) + 1), 4),
            "under": round(1.0 - _poisson_tail(total_rate, int(line) + 1), 4),
        },
        "both_teams": both,
        "calibration": correction,
    }


def predict(
    competition_code: str,
    home_team: str,
    away_team: str,
    odds: tuple[float, float, float] | None = None,
    row: pl.DataFrame | None = None,
    fixture_date: str | None = None,
) -> dict:
    """Forecast a fixture.

    `row` lets a caller supply the feature row explicitly. Backtests MUST do
    this: features must be the ones that existed before that match kicked off.
    Predicting a played match from today's form would feed the model the result
    it is being asked to predict, plus everything that happened afterwards.
    """
    if home_team == away_team:
        raise ValueError("a team cannot play itself")

    supplied_row = row is not None
    bundle = _bundle()
    if row is None and (home_team not in teams(competition_code) or away_team not in teams(competition_code)):
        raise ValueError("unknown team or competition")
    if row is None:
        row = _row_for(competition_code, home_team, away_team, fixture_date)

    have_odds = odds is not None and all(o is not None and np.isfinite(o) and o > 1.0 for o in odds)
    # Each goal-model family carries the Dixon-Coles rho fitted for it.
    rho = float(
        bundle.get("rho_market", 0.0) if have_odds else bundle.get("rho", 0.0)
    )

    if have_odds:
        inv = np.array([1.0 / odds[0], 1.0 / odds[1], 1.0 / odds[2]])
        inv = inv / inv.sum()
        row = row.with_columns(
            pl.lit(float(inv[0])).alias("mkt_home"),
            pl.lit(float(inv[1])).alias("mkt_draw"),
            pl.lit(float(inv[2])).alias("mkt_away"),
        )
        gh, ga = bundle["home_goals"], bundle["away_goals"]
        gv = _vector(row, bundle["columns"])
    else:
        gh, ga = bundle["home_goals_form"], bundle["away_goals_form"]
        gv = _vector(row, bundle["form_columns"])

    lam_h = float(np.clip(gh.predict(gv)[0], 0.05, 8.0))
    lam_a = float(np.clip(ga.predict(gv)[0], 0.05, 8.0))

    # Shift the rates by whatever this competition's settled forecasts say the
    # model has been getting wrong, before anything is derived from them — so
    # the win probabilities, the over/under and the scoreline all move together
    # instead of a corrected total contradicting an uncorrected grid.
    calibration_table = calibration.load()
    goal_correction = calibration.correction_for("goals", competition_code, calibration_table)
    if goal_correction:
        lam_h *= float(goal_correction["factor"])
        lam_a *= float(goal_correction["factor"])

    grid = tune.poisson_grid(np.array([lam_h]), np.array([lam_a]))
    if rho:
        grid = tune.dixon_coles(grid, np.array([lam_h]), np.array([lam_a]), rho)

    p1x2 = tune.grid_to_1x2(grid)[0]           # [A, D, H]
    pmap = {"A": float(p1x2[0]), "D": float(p1x2[1]), "H": float(p1x2[2])}
    basis = "form model (Dixon-Coles corrected); no closing odds for this fixture"

    if have_odds:
        basis = "odds-informed goal model (Dixon-Coles corrected)"

    g = grid[0]
    hi, ai = np.unravel_index(g.argmax(), g.shape)
    lam_h = float((g.sum(axis=1) * np.arange(g.shape[0])).sum())
    lam_a = float((g.sum(axis=0) * np.arange(g.shape[1])).sum())
    ph0 = g[0, :].sum()   # home fails to score
    pa0 = g[:, 0].sum()   # away fails to score
    btts = float(1.0 - ph0 - pa0 + g[0, 0])
    totals = np.array([
        g[(np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1])) >= k) if k == 7 else (np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1])) == k)].sum()
        for k in range(0, 8)
    ])

    def over(line: float) -> float:
        n = g.shape[0]
        mask = np.add.outer(np.arange(n), np.arange(n)) > line
        return float(g[mask].sum())

    top_indices = np.argsort(-g.ravel(), kind="stable")[:3]
    top_scores = [{"home": int(i // g.shape[1]), "away": int(i % g.shape[1]),
                   "probability": round(float(g.ravel()[i]), 4)} for i in top_indices]
    details = row.to_dicts()[0]
    training = {} if supplied_row else _training_metadata()
    context = {
        "forecast_date": fixture_date or dt.date.today().isoformat(),
        "model_trained_through": bundle.get("trained_through") or training.get("last_training_result"),
        "home_last_result": details.get("home_last_match_date"),
        "away_last_result": details.get("away_last_match_date"),
        "home_prior_matches": max(0, int(details.get("home_team_match_no") or 1) - 1),
        "away_prior_matches": max(0, int(details.get("away_team_match_no") or 1) - 1),
        "home_season_matches": details.get("home_season_matches"),
        "away_season_matches": details.get("away_season_matches"),
    }
    count_vector = _vector(row, bundle["form_columns"])
    corners = _count_market("corners", bundle, count_vector, competition_code,
                            home_team, away_team, calibration_table)
    sot = _count_market("sot", bundle, count_vector, competition_code,
                        home_team, away_team, calibration_table)

    forecast = {
        "competition_code": competition_code,
        "home_team": home_team,
        "away_team": away_team,
        "basis": basis,
        "result_pick": ["away", "draw", "home"][int(p1x2.argmax())],
        "top_scores": top_scores,
        "context": context,
        "win": {
            "home": round(pmap["H"], 4),
            "draw": round(pmap["D"], 4),
            "away": round(pmap["A"], 4),
        },
        "expected_goals": {
            "home": round(lam_h, 3),
            "away": round(lam_a, 3),
            "total": round(lam_h + lam_a, 3),
        },
        "total_goals": {
            "over_1_5": round(over(1.5), 4),
            "over_2_5": round(over(2.5), 4),
            "under_2_5": round(1.0 - over(2.5), 4),
            "over_3_5": round(over(3.5), 4),
            "btts": round(btts, 4),
            "distribution": [
                {"goals": int(k), "probability": round(float(totals[k]), 4)}
                for k in range(0, 8)
            ],
        },
        "most_likely_score": {
            "home": int(hi), "away": int(ai), "probability": round(float(g[hi, ai]), 4),
        },
        "corners": corners,
        "sot": sot,
        "calibration": {
            "goals": goal_correction,
            "corners": corners.get("calibration"),
            "sot": sot.get("calibration"),
        },
    }
    forecast["reasons"] = reasons.build(forecast, details)
    forecast["limitations"] = reasons.limits(forecast, details)

    # Log the forecast so the calibration loop can grade it once the match is
    # played. Backtests pass their own row and must not pollute the live record.
    if not supplied_row:
        calibration.record({
            "logged": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "competition_code": competition_code,
            "date": forecast["context"]["forecast_date"],
            "home_team": home_team,
            "away_team": away_team,
            "predicted": {
                "goals": forecast["expected_goals"]["total"],
                "corners": corners["total"]["expected"] if corners.get("available") else None,
                "sot": sot["total"]["expected"] if sot.get("available") else None,
            },
        })
    return forecast


def metrics(competition_code: str | None = None) -> dict:
    import json
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        raise ModelsUnavailable("No metrics — models have not been trained yet.")
    report = json.loads(path.read_text())
    if competition_code is not None:
        artifact_path = MODEL_DIR / "backtest.json"
        if not artifact_path.exists():
            raise ModelsUnavailable("Held-out predictions unavailable. Retrain the corrected models.")
        rows = [r for r in json.loads(artifact_path.read_text())["rows"]
                if r["competition_code"] == competition_code]
        summary = {"competition_code": competition_code, "n": len(rows),
                   "accuracy": None, "logloss": None, "brier_score": None,
                   "from": None, "through": None}
        if rows:
            probabilities = np.array([[r["win"][k] for k in ("away", "draw", "home")] for r in rows])
            targets = np.array([[r["actual_result"] == k for k in ("away", "draw", "home")] for r in rows])
            summary.update(
                accuracy=float((probabilities.argmax(1) == targets.argmax(1)).mean()),
                logloss=float(-(targets * np.log(np.clip(probabilities, 1e-12, 1))).sum(1).mean()),
                brier_score=float(((probabilities - targets) ** 2).sum(1).mean()),
                **{"from": min(r["date"] for r in rows), "through": max(r["date"] for r in rows)},
            )
        report["competition_metrics"] = summary
    return report


def recent_backtest(competition_code: str, limit: int = 10) -> list[dict]:
    """Read frozen predictions made before the final production refit."""
    import json
    path = MODEL_DIR / "backtest.json"
    if not path.exists():
        raise ModelsUnavailable("Held-out predictions unavailable. Retrain the corrected models.")
    artifact = json.loads(path.read_text())
    rows = [r for r in artifact["rows"] if r["competition_code"] == competition_code]
    return sorted(rows, key=lambda r: r["date"], reverse=True)[:limit]
