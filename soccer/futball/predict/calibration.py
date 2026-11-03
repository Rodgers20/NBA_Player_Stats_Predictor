"""Correct the model's own drift from its own past misses.

A rate model can be well-fitted and still sit consistently on one side of the
truth: it settles on 9.8 corners a game while the division has drifted to 10.6,
and every forecast leans under. Nothing in the fit notices, because the fit
already minimised its loss on the data it saw.

This module is the part that notices. For each competition and market it
compares what was **predicted before kick-off** against what happened, and
derives one multiplicative correction per rate:

    correction = mean(actual) / mean(predicted)

Two rules keep it a correction rather than a second, unvalidated model:

* **Only out-of-sample predictions count.** The frozen held-out window from
  training, plus the live forecasts this app served (`served.jsonl`) once their
  results are in the lake. Comparing against matches the production model was
  refitted on would measure a bias it had already absorbed and report none.
* **It is clipped, and it needs evidence.** Below `MIN_MATCHES` nothing is
  applied at all, and the correction is bounded to ±`MAX_SHIFT`. A run of ten
  high-scoring games is noise; a persistent 8% lean across two hundred is drift.

The served payload carries the correction that was applied, so a shifted number
is never presented as the raw model's opinion.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, asdict

import polars as pl

from .. import config
from ..storage import write_text_atomic
from .train import MODEL_DIR

log = logging.getLogger(__name__)

#: Where live forecasts are appended as they are served.
SERVED_LOG = MODEL_DIR / "served.jsonl"
CALIBRATION_PATH = MODEL_DIR / "calibration.json"

#: Fewer settled predictions than this and any apparent bias is noise.
MIN_MATCHES = 60

#: The most a correction may move a rate. Beyond this the model is wrong in a
#: way a scalar cannot fix, and quietly stretching it would hide that.
MAX_SHIFT = 0.20

#: Markets and the lake columns holding what actually happened.
MARKETS: dict[str, tuple[str, str]] = {
    "goals": ("home_goals", "away_goals"),
    "corners": ("home_corners", "away_corners"),
    "sot": ("home_sot", "away_sot"),
}


@dataclass(frozen=True)
class Correction:
    market: str
    competition_code: str
    n: int
    predicted_mean: float
    actual_mean: float
    #: What the rate is multiplied by. 1.0 means the model needs no help.
    factor: float
    clipped: bool
    source: str
    through: str | None

    @property
    def lean(self) -> str:
        if self.factor > 1.005:
            return "under"
        if self.factor < 0.995:
            return "over"
        return "none"


def _clip(factor: float) -> tuple[float, bool]:
    low, high = 1.0 - MAX_SHIFT, 1.0 + MAX_SHIFT
    if factor < low:
        return low, True
    if factor > high:
        return high, True
    return factor, False


def _from_backtest() -> list[dict]:
    """Held-out predictions frozen by training, one row per match."""
    path = MODEL_DIR / "backtest.json"
    if not path.exists():
        return []
    rows = []
    for r in json.loads(path.read_text()).get("rows", []):
        entry = {
            "competition_code": r["competition_code"], "date": r["date"],
            "source": "held-out",
            "goals": (r.get("expected_total"), r.get("actual_total")),
        }
        for market, payload in (r.get("counts") or {}).items():
            entry[market] = (
                payload["predicted_home"] + payload["predicted_away"],
                payload["actual_home"] + payload["actual_away"],
            )
        rows.append(entry)
    return rows


def _from_served() -> list[dict]:
    """Live forecasts, matched against results that have since landed."""
    if not SERVED_LOG.exists():
        return []

    served: list[dict] = []
    for line in SERVED_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            served.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a truncated final write must not poison the whole log
    if not served:
        return []

    results_path = config.SILVER / "results" / "results.parquet"
    if not results_path.exists():
        return []
    lake = pl.read_parquet(results_path)

    actual: dict[tuple, dict] = {}
    for r in lake.select(
        ["competition_code", "date", "home_team", "away_team", *sum(MARKETS.values(), ())]
    ).to_dicts():
        actual[(r["competition_code"], str(r["date"]), r["home_team"], r["away_team"])] = r

    rows = []
    for s in served:
        key = (s.get("competition_code"), s.get("date"), s.get("home_team"), s.get("away_team"))
        played = actual.get(key)
        if played is None:
            continue  # not yet played, or the fixture moved
        entry = {"competition_code": key[0], "date": key[1], "source": "served"}
        for market, (home_col, away_col) in MARKETS.items():
            predicted = s.get("predicted", {}).get(market)
            if predicted is None:
                continue
            if played[home_col] is None or played[away_col] is None:
                continue
            entry[market] = (float(predicted), float(played[home_col] + played[away_col]))
        rows.append(entry)
    return rows


def measure() -> dict[str, dict[str, Correction]]:
    """One correction per market per competition, from settled predictions."""
    rows = _from_backtest() + _from_served()
    out: dict[str, dict[str, Correction]] = {m: {} for m in MARKETS}

    for market in MARKETS:
        by_code: dict[str, list[tuple[float, float, str, str]]] = {}
        for r in rows:
            pair = r.get(market)
            if not pair or pair[0] is None or pair[1] is None:
                continue
            by_code.setdefault(r["competition_code"], []).append(
                (float(pair[0]), float(pair[1]), r["source"], r["date"])
            )

        for code, samples in by_code.items():
            if len(samples) < MIN_MATCHES:
                continue
            predicted = sum(s[0] for s in samples) / len(samples)
            observed = sum(s[1] for s in samples) / len(samples)
            if predicted <= 0:
                continue
            factor, clipped = _clip(observed / predicted)
            sources = sorted({s[2] for s in samples})
            out[market][code] = Correction(
                market=market, competition_code=code, n=len(samples),
                predicted_mean=round(predicted, 3), actual_mean=round(observed, 3),
                factor=round(factor, 4), clipped=clipped,
                source=" + ".join(sources),
                through=max(s[3] for s in samples),
            )
    return out


def refresh() -> dict:
    """Recompute and persist. Called by the refresh pipeline."""
    measured = measure()
    payload = {
        "generated": dt.date.today().isoformat(),
        "min_matches": MIN_MATCHES,
        "max_shift": MAX_SHIFT,
        "markets": {
            market: {code: asdict(c) for code, c in codes.items()}
            for market, codes in measured.items()
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    write_text_atomic(CALIBRATION_PATH, json.dumps(payload, indent=2))
    return payload


def load() -> dict:
    if not CALIBRATION_PATH.exists():
        return {"markets": {}}
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("calibration.json unreadable; serving uncorrected rates")
        return {"markets": {}}


def correction_for(market: str, competition_code: str, table: dict | None = None) -> dict | None:
    """The stored correction, or None when there is not enough evidence yet."""
    table = load() if table is None else table
    return (table.get("markets", {}).get(market) or {}).get(competition_code)


def _served_key(row: dict) -> tuple:
    """One forecast per fixture per day — the unit the grader should weigh."""
    return (row.get("competition_code"), row.get("date"),
            row.get("home_team"), row.get("away_team"))


#: Keys already on disk, so a fixture is logged once a day however often it is
#: requested. Loaded lazily; `None` means "not read yet".
_logged: set[tuple] | None = None


def _known_keys() -> set[tuple]:
    global _logged
    if _logged is None:
        seen = set()
        if SERVED_LOG.exists():
            for line in SERVED_LOG.read_text().splitlines():
                try:
                    seen.add(_served_key(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        _logged = seen
    return _logged


def record(prediction: dict) -> None:
    """Append a served forecast so its accuracy can be measured later.

    **Deduplicated per fixture per day.** The matchday board forecasts ten
    fixtures the moment the page opens, so without this a single afternoon of
    browsing would write the same forecast dozens of times and the grader would
    weigh that matchday dozens of times over — an accidental thumb on the scale
    of the model's own correction.

    Best-effort: a forecast that cannot be logged must never break the response
    the user asked for.
    """
    key = _served_key(prediction)
    if key in _known_keys():
        return
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with SERVED_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(prediction, separators=(",", ":")) + "\n")
        _known_keys().add(key)
    except OSError as exc:  # noqa: BLE001 - logging is not the point of the request
        log.debug("could not log served prediction: %s", exc)


def report() -> int:
    payload = refresh()
    print("\n  PREDICTION CALIBRATION")
    print("  " + "-" * 66)
    any_row = False
    for market, codes in payload["markets"].items():
        for code, c in sorted(codes.items()):
            any_row = True
            arrow = "→" if abs(c["factor"] - 1) < 0.005 else ("↑" if c["factor"] > 1 else "↓")
            print(f"  {market:<8} {code:<5} n={c['n']:<5} predicted {c['predicted_mean']:>6.2f}"
                  f"   actual {c['actual_mean']:>6.2f}   {arrow} x{c['factor']:.3f}"
                  f"{'  (clipped)' if c['clipped'] else ''}")
    if not any_row:
        print(f"  Nothing corrected yet — every market has under {MIN_MATCHES} settled")
        print("  out-of-sample predictions.")
    print("  " + "-" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
