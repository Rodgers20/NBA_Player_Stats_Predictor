"""Track WNBA prop predictions over time and grade them once games conclude.

Storage: data/wnba/prediction_history.json
Format:
    {
      "YYYY-MM-DD": [
        {
          "player_name": "...", "team": "...", "stat": "...",
          "line": 22.5, "pick": "OVER", "projected": 24.1,
          "hit_prob": 0.7, "ev": 0.15, "confidence": "HIGH",
          "actual": null,        # populated after grading
          "outcome": null,       # "WIN" | "LOSS" | "PUSH" after grading
          "graded_at": null,
        },
        ...
      ],
      ...
    }
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from utils.league_config import get_config

logger = logging.getLogger(__name__)

_HISTORY_FILE = get_config("wnba").data_dir / "prediction_history.json"

# Combo stat -> component stats (kept here rather than importing from
# wnba_props to keep this module independent of the odds/props pipeline)
_COMBO_COMPONENTS = {
    "PTS+REB": ["PTS", "REB"],
    "PTS+AST": ["PTS", "AST"],
    "PTS+REB+AST": ["PTS", "REB", "AST"],
}


def _load_history() -> dict:
    if not _HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(_HISTORY_FILE.read_text() or "{}")
    except Exception as e:
        logger.warning(f"[WNBA-Tracker] history read failed: {e}")
        return {}


def _save_history(history: dict) -> None:
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_FILE.write_text(json.dumps(history, indent=2))


def record_predictions(props: Iterable, target_date: str | None = None) -> int:
    """Snapshot today's props to history. No-op if the date already has entries.

    Returns count of props recorded.
    """
    day = target_date or date.today().isoformat()
    history = _load_history()

    if day in history and history[day]:
        # Already snapshotted today — skip to avoid overwriting with stale data
        logger.info(f"[WNBA-Tracker] {day} already recorded ({len(history[day])} entries)")
        return 0

    entries = []
    for p in props:
        entries.append({
            "player_name": p.player_name,
            "team": p.team,
            "stat": p.stat,
            "line": p.line,
            "pick": p.pick,
            "projected": round(p.projected, 2),
            "hit_prob": round(p.hit_prob, 3),
            "ev": round(p.ev, 3),
            "confidence": p.confidence,
            "over_price": p.over_price,
            "under_price": p.under_price,
            "bookmaker": p.bookmaker,
            "actual": None,
            "outcome": None,
            "graded_at": None,
        })
    history[day] = entries
    _save_history(history)
    logger.info(f"[WNBA-Tracker] Recorded {len(entries)} predictions for {day}")
    return len(entries)


def grade_date(target_date: str, wnba_df) -> int:
    """Grade predictions for target_date against actual game logs.

    Returns count of newly graded predictions.
    """
    history = _load_history()
    entries = history.get(target_date, [])
    if not entries:
        return 0

    day_games = wnba_df[wnba_df["_date"].dt.date == datetime.fromisoformat(target_date).date()]
    if day_games.empty:
        logger.info(f"[WNBA-Tracker] No game log data for {target_date} yet — skipping grade")
        return 0

    now = datetime.now(timezone.utc).isoformat()
    graded = 0
    for entry in entries:
        if entry.get("outcome") is not None:
            continue

        player_rows = day_games[day_games["PLAYER_NAME"] == entry["player_name"]]
        if player_rows.empty:
            entry["outcome"] = "DNP"     # Did Not Play
            entry["graded_at"] = now
            graded += 1
            continue

        row = player_rows.iloc[0]
        components = _COMBO_COMPONENTS.get(entry["stat"], [entry["stat"]])
        try:
            actual = float(sum(row.get(c, 0) or 0 for c in components))
        except Exception:
            continue
        entry["actual"] = actual

        line = float(entry["line"])
        if actual == line:
            entry["outcome"] = "PUSH"
        elif entry["pick"] == "OVER":
            entry["outcome"] = "WIN" if actual > line else "LOSS"
        else:
            entry["outcome"] = "WIN" if actual < line else "LOSS"
        entry["graded_at"] = now
        graded += 1

    _save_history(history)
    logger.info(f"[WNBA-Tracker] Graded {graded} entries for {target_date}")
    return graded


def grade_pending(wnba_df, max_lookback_days: int = 7) -> int:
    """Grade any past dates with ungraded entries in the last N days."""
    history = _load_history()
    total = 0
    today = date.today()
    for day_str in sorted(history.keys()):
        try:
            day = datetime.fromisoformat(day_str).date()
        except ValueError:
            continue
        if day >= today:
            continue
        if (today - day).days > max_lookback_days:
            continue
        ungraded = [e for e in history[day_str] if e.get("outcome") is None]
        if ungraded:
            total += grade_date(day_str, wnba_df)
    return total


_CALIBRATION_FILE = get_config("wnba").data_dir / "model_calibration.json"


def compute_calibration_offsets(min_samples: int = 25) -> dict:
    """Compute per-stat bias offsets: how much the model over/under-predicts on average.

    Positive offset means the model over-predicts (subtract from future predictions).
    Requires at least `min_samples` graded entries per stat for a stable estimate.
    Saves result to data/wnba/model_calibration.json.
    """
    history = _load_history()
    diffs: dict[str, list[float]] = {}

    for _day, entries in history.items():
        for e in entries:
            if e.get("outcome") not in ("WIN", "LOSS", "PUSH"):
                continue
            actual = e.get("actual")
            projected = e.get("projected")
            if actual is None or projected is None:
                continue
            diffs.setdefault(e["stat"], []).append(float(projected) - float(actual))

    offsets: dict[str, float] = {}
    for stat, arr in diffs.items():
        if len(arr) >= min_samples:
            offsets[stat] = round(sum(arr) / len(arr), 3)

    if offsets:
        _CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CALIBRATION_FILE.write_text(json.dumps(offsets, indent=2))
        logger.info(f"[WNBA-Tracker] Calibration offsets: {offsets}")

    return offsets


def get_calibration_offsets() -> dict:
    """Read stored per-stat calibration offsets. Empty dict if none yet."""
    if not _CALIBRATION_FILE.exists():
        return {}
    try:
        return json.loads(_CALIBRATION_FILE.read_text() or "{}")
    except Exception:
        return {}


def get_accuracy_summary(lookback_days: int = 30) -> dict:
    """Return system record: wins / losses / pushes / hit rate over recent history."""
    history = _load_history()
    today = date.today()
    wins = losses = pushes = dnp = 0
    high_wins = high_losses = 0
    parlay_ready = 0

    for day_str, entries in history.items():
        try:
            day = datetime.fromisoformat(day_str).date()
        except ValueError:
            continue
        if (today - day).days > lookback_days:
            continue
        for e in entries:
            outcome = e.get("outcome")
            if outcome == "WIN":
                wins += 1
                if e.get("confidence") == "HIGH":
                    high_wins += 1
            elif outcome == "LOSS":
                losses += 1
                if e.get("confidence") == "HIGH":
                    high_losses += 1
            elif outcome == "PUSH":
                pushes += 1
            elif outcome == "DNP":
                dnp += 1
            else:
                parlay_ready += 1

    total_graded = wins + losses
    hit_rate = wins / total_graded if total_graded else 0.0
    high_total = high_wins + high_losses
    high_rate = high_wins / high_total if high_total else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "dnp": dnp,
        "pending": parlay_ready,
        "hit_rate": hit_rate,
        "high_wins": high_wins,
        "high_losses": high_losses,
        "high_hit_rate": high_rate,
        "lookback_days": lookback_days,
    }
