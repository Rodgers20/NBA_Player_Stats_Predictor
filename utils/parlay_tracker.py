"""
parlay_tracker.py
=================
Tracks generated parlays, grades them after games complete, and computes P&L.

Storage: data/parlays_history.json

Schema per date entry:
{
    "generated_at": ISO timestamp,
    "graded_at":    ISO timestamp | null,
    "parlays": [
        {
            "label":        "3-Leg Parlay #1",
            "type":         "three_leg" | "two_leg" | "alt" | "over" | "under" | "defense" | "ml",
            "legs": [
                {
                    "player":    str,
                    "stat":      str,
                    "direction": "Over" | "Under",
                    "line":      float,
                    "actual":    float | null,   # filled at grade time
                    "hit":       bool | null,    # filled at grade time
                }
            ],
            "odds_american": int,
            "odds_decimal":  float,
            "win_prob":      float,   # model's combined win %
            "result":        "win" | "loss" | "pending" | null,
            "profit_10":     float | null,   # profit on a $10 stake
        }
    ],
    "summary": {
        "total":    int,
        "wins":     int,
        "losses":   int,
        "pending":  int,
        "profit_10": float,
        "roi_pct":   float,
    }
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from utils.league_config import get_config
DATA_DIR      = get_config("nba").data_dir
PARLAYS_FILE  = DATA_DIR / "parlays_history.json"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load() -> dict:
    if PARLAYS_FILE.exists():
        try:
            return json.loads(PARLAYS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save(history: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARLAYS_FILE.write_text(json.dumps(history, indent=2, default=str))


# ── Save ──────────────────────────────────────────────────────────────────────

def save_daily_parlays(date: str, parlays: dict) -> None:
    """Persist today's generated parlays before games start.

    Args:
        date:    "YYYY-MM-DD"
        parlays: Output of build_all_parlays() — dict with keys
                 best_bets, two_leg, three_leg, ml, alt, over, under, defense.
    """
    history = _load()

    # Never overwrite a graded record
    if date in history and history[date].get("graded_at"):
        return

    entries: list[dict] = []

    def _add_parlay(p: dict) -> None:
        legs_stripped = [
            {
                "player":    leg["player"],
                "stat":      leg.get("stat", ""),
                "direction": leg.get("direction", "Over"),
                "line":      float(leg.get("line") or leg.get("threshold") or 0),
                "actual":    None,
                "hit":       None,
            }
            for leg in p.get("legs", [])
        ]
        odds = p.get("odds", {})
        entries.append({
            "label":        p.get("label", "Parlay"),
            "type":         p.get("type", "unknown"),
            "legs":         legs_stripped,
            "odds_american": odds.get("american", 0),
            "odds_decimal":  odds.get("decimal", 1.0),
            "win_prob":      odds.get("win_prob", 0.0),
            "result":        None,
            "profit_10":     None,
        })

    # Collect all parlay types — handle both old and new dict shapes
    for key in ("over", "pts", "reb", "ast", "combo",
                "ml", "spread", "totals", "alt_over", "reduced",
                "two_leg", "three_leg", "alt", "under", "defense"):
        val = parlays.get(key)
        if not val:
            continue
        # ml used to be a single dict; now it's a list — handle both
        if isinstance(val, list):
            for p in val:
                _add_parlay(p)
        elif isinstance(val, dict):
            _add_parlay(val)

    if not entries:
        return

    history[date] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graded_at":    None,
        "parlays":      entries,
        "summary":      {},
    }
    _save(history)
    print(f"[ParlayTracker] Saved {len(entries)} parlays for {date}")


# ── Grade ─────────────────────────────────────────────────────────────────────

def grade_parlays(date: str) -> dict:
    """Grade saved parlays against actual game logs.

    A parlay wins only if every leg hits.
    Assumes $10 stake per parlay for P&L calculation.
    """
    history = _load()
    if date not in history:
        print(f"[ParlayTracker] No parlays saved for {date}")
        return {}

    record = history[date]
    if record.get("graded_at"):
        print(f"[ParlayTracker] {date} already graded")
        return record.get("summary", {})

    # Load actual game logs
    logs_df = _load_game_logs_for_date(date)
    if logs_df is None or logs_df.empty:
        print(f"[ParlayTracker] No game log data for {date} — cannot grade")
        return {}

    wins = losses = pending = 0
    total_profit = 0.0

    for parlay in record["parlays"]:
        all_hit      = True
        any_miss     = False
        any_pending  = False

        for leg in parlay["legs"]:
            player    = leg["player"]
            stat      = leg["stat"]
            direction = leg.get("direction", "Over")
            line      = leg.get("line", 0)

            actual = _lookup_actual_stat(logs_df, player, stat, date)
            if actual is None:
                any_pending = True
                all_hit = False
                continue

            leg["actual"] = actual
            is_hit = (actual >= line) if direction == "Over" else (actual <= line)
            leg["hit"] = is_hit

            if not is_hit:
                any_miss = True
                all_hit  = False

        if any_pending and not any_miss:
            # Some legs not yet in data — leave pending
            parlay["result"] = "pending"
            pending += 1
        elif all_hit:
            parlay["result"] = "win"
            profit = round((parlay["odds_decimal"] - 1.0) * 10, 2)
            parlay["profit_10"] = profit
            total_profit += profit
            wins += 1
        else:
            parlay["result"] = "loss"
            parlay["profit_10"] = -10.0
            total_profit -= 10.0
            losses += 1

    graded = wins + losses
    staked = graded * 10
    roi    = round(total_profit / staked * 100, 1) if staked > 0 else 0.0

    summary = {
        "total":     len(record["parlays"]),
        "wins":      wins,
        "losses":    losses,
        "pending":   pending,
        "profit_10": round(total_profit, 2),
        "roi_pct":   roi,
    }
    record["summary"]   = summary
    record["graded_at"] = datetime.now(timezone.utc).isoformat()
    _save(history)

    print(
        f"[ParlayTracker] Graded {date}: {wins}W/{losses}L "
        f"(pending: {pending})  profit: ${total_profit:+.2f}  ROI: {roi}%"
    )
    return summary


# ── Record aggregation ────────────────────────────────────────────────────────

def get_parlays_record() -> dict:
    """Return all-time parlay record across every graded date.

    Returns:
        {
            "total":         int,
            "wins":          int,
            "losses":        int,
            "win_pct":       float,
            "profit_10":     float,   # cumulative profit at $10/parlay
            "roi_pct":       float,
            "days_graded":   int,
            "recent_7d_win_pct": float,
            "by_type": {
                "two_leg":   {"wins": int, "losses": int, "profit_10": float},
                "three_leg": {...},
                ...
            }
        }
    """
    history = _load()

    total = wins = losses = 0
    profit = 0.0
    days_graded = 0
    by_type: dict[str, dict] = {}

    import datetime as _dt
    today   = _dt.date.today()
    cutoff7 = (today - _dt.timedelta(days=7)).isoformat()
    recent_wins = recent_losses = 0

    for date_str, rec in sorted(history.items()):
        days_graded += 1
        if not rec.get("graded_at"):
            continue

        for parlay in rec.get("parlays", []):
            result = parlay.get("result")
            if result not in ("win", "loss"):
                continue

            total += 1
            p_type = parlay.get("type", "unknown")
            bt = by_type.setdefault(p_type, {"wins": 0, "losses": 0, "profit_10": 0.0})

            if result == "win":
                wins += 1
                bt["wins"] += 1
                pnl = parlay.get("profit_10", 0) or 0
                profit      += pnl
                bt["profit_10"] += pnl
                if date_str >= cutoff7:
                    recent_wins += 1
            else:
                losses += 1
                bt["losses"] += 1
                profit      -= 10.0
                bt["profit_10"] -= 10.0
                if date_str >= cutoff7:
                    recent_losses += 1

    staked = total * 10
    roi    = round(profit / staked * 100, 1) if staked > 0 else 0.0

    recent_total = recent_wins + recent_losses
    recent_pct   = round(recent_wins / recent_total * 100, 1) if recent_total else 0.0

    return {
        "total":             total,
        "wins":              wins,
        "losses":            losses,
        "win_pct":           round(wins / total * 100, 1) if total else 0.0,
        "profit_10":         round(profit, 2),
        "roi_pct":           roi,
        "days_graded":       days_graded,
        "recent_7d_win_pct": recent_pct,
        "recent_7d_total":   recent_total,
        "by_type":           by_type,
    }


# ── Stat lookup (mirrors prediction_tracker.py) ───────────────────────────────

def _load_game_logs_for_date(date: str):
    try:
        import pandas as pd

        parquet    = DATA_DIR / "player_game_logs.parquet"
        engineered = DATA_DIR / "engineered_data.parquet"
        csv_path   = DATA_DIR / "player_game_logs.csv"

        if parquet.exists():
            df = pd.read_parquet(parquet)
        elif engineered.exists():
            df = pd.read_parquet(engineered)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            return None

        if "_date" not in df.columns:
            df["_date"] = pd.to_datetime(
                df["GAME_DATE"], format="%b %d, %Y", errors="coerce"
            )

        target = pd.to_datetime(date)
        return df[df["_date"].dt.date == target.date()]
    except Exception as e:
        print(f"[ParlayTracker] Could not load game logs: {e}")
        return None


def _lookup_actual_stat(df, player: str, stat: str, date: str):
    """Return actual value of `stat` for `player`. Handles combo stats."""
    try:
        rows = df[df["PLAYER_NAME"] == player]
        if rows.empty:
            return None

        if "+" in stat:
            total = 0.0
            for part in stat.split("+"):
                if part not in rows.columns:
                    return None
                total += float(rows[part].iloc[0])
            return round(total, 1)

        if stat not in rows.columns:
            return None
        return round(float(rows[stat].iloc[0]), 1)
    except Exception:
        return None
