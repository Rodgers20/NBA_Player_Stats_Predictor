"""
Prediction Tracker
==================
Save daily model predictions, grade them against actual results, and export
to Excel for ongoing model evaluation.

Storage: data/prediction_history.json
Excel:   data/prediction_report.xlsx
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_FILE = DATA_DIR / "prediction_history.json"
EXCEL_PATH = DATA_DIR / "prediction_report.xlsx"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_history() -> dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_history(history: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ─── public API ───────────────────────────────────────────────────────────────

def save_daily_predictions(date: str, games: list[dict]) -> None:
    """
    Store model predictions for the day.

    Each game dict must have at minimum:
        game_id, home, away, game_time, predictions (dict)
    """
    history = _load_history()

    # Don't overwrite graded records
    if date in history and history[date].get("graded_at"):
        return

    history[date] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graded_at": None,
        "games": [
            {
                "game_id":  g.get("game_id", f"{g.get('away','?')}@{g.get('home','?')}"),
                "home":     g.get("home", ""),
                "away":     g.get("away", ""),
                "game_time": g.get("game_time", ""),
                "predictions": g.get("predictions", {}),
                "actuals": {},
                "grades": {},
            }
            for g in games
        ],
        "summary": {},
    }
    _save_history(history)
    print(f"[Tracker] Saved {len(games)} predictions for {date}")


def grade_predictions(date: str) -> dict:
    """
    Fetch final scores from ESPN and grade stored predictions.
    Returns the updated summary dict. Safe to call even if already graded.
    """
    history = _load_history()
    if date not in history:
        print(f"[Tracker] No predictions found for {date}")
        return {}

    record = history[date]
    if record.get("graded_at"):
        print(f"[Tracker] {date} already graded")
        return record.get("summary", {})

    # Fetch actual scores from ESPN
    actuals = _fetch_actuals(date)
    if not actuals:
        print(f"[Tracker] No final scores available yet for {date}")
        return {}

    ml_correct = ml_total = 0
    sp_correct = sp_total = 0
    ou_correct = ou_total = 0

    for game in record["games"]:
        key = game["game_id"]
        result = actuals.get(key) or actuals.get(f"{game['away']}@{game['home']}")
        if not result:
            continue

        hs = result.get("home_score")
        as_ = result.get("away_score")
        if hs is None or as_ is None:
            continue

        game["actuals"] = {
            "home_score": hs,
            "away_score": as_,
            "winner": game["home"] if hs > as_ else game["away"],
            "total": hs + as_,
        }

        preds = game.get("predictions", {})
        grades: dict[str, Any] = {}

        # Moneyline
        if preds.get("winner_pick"):
            ml_total += 1
            winner_correct = preds["winner_pick"] == game["actuals"]["winner"]
            grades["winner_correct"] = winner_correct
            if winner_correct:
                ml_correct += 1

        # Spread
        if preds.get("spread_pick") and preds.get("spread_line") is not None:
            sp_total += 1
            margin = hs - as_
            line = float(preds["spread_line"])
            # spread_pick = "HOME" or "AWAY"
            if preds["spread_pick"] == "HOME":
                covered = margin > (-line)
            else:
                covered = (-margin) > (-line)
            grades["spread_correct"] = covered
            if covered:
                sp_correct += 1

        # Over/Under
        if preds.get("total_pick") and preds.get("total_line") is not None:
            ou_total += 1
            total_actual = hs + as_
            total_line = float(preds["total_line"])
            ou_correct_flag = (
                (preds["total_pick"] == "OVER" and total_actual > total_line)
                or (preds["total_pick"] == "UNDER" and total_actual < total_line)
            )
            grades["total_correct"] = ou_correct_flag
            if ou_correct_flag:
                ou_correct += 1

        game["grades"] = grades

    def _pct(c: int, t: int) -> float:
        return round(c / t * 100, 1) if t else 0.0

    summary = {
        "moneyline": {"correct": ml_correct, "total": ml_total, "pct": _pct(ml_correct, ml_total)},
        "spread":    {"correct": sp_correct, "total": sp_total, "pct": _pct(sp_correct, sp_total)},
        "total":     {"correct": ou_correct, "total": ou_total, "pct": _pct(ou_correct, ou_total)},
    }
    record["summary"] = summary
    record["graded_at"] = datetime.now(timezone.utc).isoformat()
    _save_history(history)
    print(f"[Tracker] Graded {date}: ML {summary['moneyline']['pct']}% | "
          f"Spread {summary['spread']['pct']}% | O/U {summary['total']['pct']}%")
    return summary


def export_to_excel(output_path: str | None = None) -> str:
    """
    Export full prediction history to Excel with 4 sheets:
      - Summary:    one row per date
      - Moneyline:  per-game ML picks + results
      - Spread:     per-game spread picks + results
      - Total (O/U): per-game O/U picks + results

    Returns the path to the written file.
    """
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise RuntimeError("openpyxl is required for Excel export. Run: pip install openpyxl")

    history = _load_history()
    if not history:
        raise ValueError("No prediction history found")

    path = Path(output_path) if output_path else EXCEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # remove default sheet

    green = PatternFill("solid", fgColor="C6EFCE")
    red   = PatternFill("solid", fgColor="FFC7CE")
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF")
    center = Alignment(horizontal="center")

    def _make_sheet(title: str, headers: list[str]) -> Any:
        ws = wb.create_sheet(title)
        ws.append(headers)
        for col_idx, _ in enumerate(headers, 1):
            cell = ws.cell(1, col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center
        return ws

    def _set_col_widths(ws: Any, widths: list[int]) -> None:
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    ws_sum = _make_sheet("Summary", [
        "Date", "ML Correct", "ML Total", "ML %",
        "Spread Correct", "Spread Total", "Spread %",
        "O/U Correct", "O/U Total", "O/U %", "Overall %",
    ])
    _set_col_widths(ws_sum, [12, 11, 9, 8, 14, 12, 10, 11, 9, 8, 10])

    for date in sorted(history.keys()):
        rec = history[date]
        s = rec.get("summary", {})
        ml  = s.get("moneyline", {})
        sp  = s.get("spread", {})
        ou  = s.get("total", {})
        all_c = ml.get("correct", 0) + sp.get("correct", 0) + ou.get("correct", 0)
        all_t = ml.get("total", 0)   + sp.get("total", 0)   + ou.get("total", 0)
        overall = round(all_c / all_t * 100, 1) if all_t else 0.0
        row = [
            date,
            ml.get("correct", ""), ml.get("total", ""), ml.get("pct", ""),
            sp.get("correct", ""), sp.get("total", ""), sp.get("pct", ""),
            ou.get("correct", ""), ou.get("total", ""), ou.get("pct", ""),
            overall,
        ]
        ws_sum.append(row)

    # ── Sheet 2: Moneyline ────────────────────────────────────────────────────
    ws_ml = _make_sheet("Moneyline", [
        "Date", "Game", "Pick", "Confidence",
        "Model Home", "Model Away", "Actual Home", "Actual Away",
        "Result", "Correct",
    ])
    _set_col_widths(ws_ml, [12, 14, 8, 12, 12, 11, 12, 11, 10, 9])

    # ── Sheet 3: Spread ───────────────────────────────────────────────────────
    ws_sp = _make_sheet("Spread", [
        "Date", "Game", "Pick", "Book Line", "Model Line", "Edge",
        "Actual Home", "Actual Away", "Result", "Correct",
    ])
    _set_col_widths(ws_sp, [12, 14, 10, 10, 11, 8, 12, 11, 10, 9])

    # ── Sheet 4: Total (O/U) ──────────────────────────────────────────────────
    ws_ou = _make_sheet("Total (O/U)", [
        "Date", "Game", "Pick", "Book Line", "Model Total", "Edge",
        "Actual Total", "Result", "Correct",
    ])
    _set_col_widths(ws_ou, [12, 14, 8, 10, 12, 8, 13, 10, 9])

    def _result_fill(correct: bool | None) -> PatternFill | None:
        if correct is True:
            return green
        if correct is False:
            return red
        return None

    for date in sorted(history.keys()):
        for g in history[date].get("games", []):
            game_label = f"{g.get('away', '?')} @ {g.get('home', '?')}"
            preds  = g.get("predictions", {})
            actual = g.get("actuals", {})
            grades = g.get("grades", {})
            hs = actual.get("home_score", "")
            as_ = actual.get("away_score", "")

            # Moneyline row
            ml_pick   = preds.get("winner_pick", "")
            ml_conf   = preds.get("winner_confidence", "")
            ml_mh     = preds.get("model_home_score", "")
            ml_ma     = preds.get("model_away_score", "")
            ml_result = actual.get("winner", "")
            ml_cor    = grades.get("winner_correct")
            ml_row = ws_ml.append([
                date, game_label, ml_pick, ml_conf,
                ml_mh, ml_ma, hs, as_,
                ml_result, "✓" if ml_cor is True else ("✗" if ml_cor is False else ""),
            ])
            if ml_cor is not None:
                f = _result_fill(ml_cor)
                if f:
                    for col in range(1, 11):
                        ws_ml.cell(ws_ml.max_row, col).fill = f

            # Spread row
            sp_pick   = f"{preds.get('spread_pick','')} {preds.get('spread_team','')}"
            sp_line   = preds.get("spread_line", "")
            model_mar = ""
            if preds.get("model_home_score") and preds.get("model_away_score"):
                model_mar = round(float(preds["model_home_score"]) - float(preds["model_away_score"]), 1)
            sp_edge = ""
            try:
                sp_edge = round(float(model_mar) - float(sp_line), 1) if sp_line != "" else ""
            except Exception:
                pass
            sp_cor = grades.get("spread_correct")
            ws_sp.append([
                date, game_label, sp_pick.strip(), sp_line, model_mar, sp_edge,
                hs, as_,
                f"{hs}-{as_}" if hs != "" else "",
                "✓" if sp_cor is True else ("✗" if sp_cor is False else ""),
            ])
            if sp_cor is not None:
                f = _result_fill(sp_cor)
                if f:
                    for col in range(1, 11):
                        ws_sp.cell(ws_sp.max_row, col).fill = f

            # O/U row
            ou_pick   = preds.get("total_pick", "")
            ou_line   = preds.get("total_line", "")
            ou_model  = preds.get("model_total", "")
            ou_edge = ""
            try:
                ou_edge = round(float(ou_model) - float(ou_line), 1) if ou_line != "" else ""
            except Exception:
                pass
            ou_actual = actual.get("total", "")
            ou_cor    = grades.get("total_correct")
            ws_ou.append([
                date, game_label, ou_pick, ou_line, ou_model, ou_edge,
                ou_actual,
                f"{ou_actual} ({'O' if ou_cor else 'U'})" if ou_cor is not None and ou_actual != "" else "",
                "✓" if ou_cor is True else ("✗" if ou_cor is False else ""),
            ])
            if ou_cor is not None:
                f = _result_fill(ou_cor)
                if f:
                    for col in range(1, 10):
                        ws_ou.cell(ws_ou.max_row, col).fill = f

    wb.save(path)
    print(f"[Tracker] Excel exported to {path}")
    return str(path)


# ─── ESPN score fetcher ────────────────────────────────────────────────────────

def _fetch_actuals(date: str) -> dict[str, dict]:
    """
    Fetch final scores from ESPN for the given date.
    Returns {game_id: {"home_score": int, "away_score": int}} or empty dict.
    """
    try:
        import requests
        from utils.data_fetch import _normalize_espn_abbrev  # type: ignore
        espn_date = date.replace("-", "")
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        resp = requests.get(url, params={"dates": espn_date}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results: dict[str, dict] = {}
        for event in data.get("events", []):
            status = event.get("status", {}).get("type", {}).get("description", "")
            if "final" not in status.lower():
                continue  # skip in-progress or future games
            comp = event["competitions"][0]
            home_score = away_score = None
            home_abbr = away_abbr = ""
            for competitor in comp["competitors"]:
                abbr = _normalize_espn_abbrev(competitor["team"]["abbreviation"])
                score = int(competitor.get("score", 0))
                if competitor["homeAway"] == "home":
                    home_abbr = abbr
                    home_score = score
                else:
                    away_abbr = abbr
                    away_score = score
            if home_score is not None and away_score is not None:
                game_id = f"{away_abbr}@{home_abbr}"
                results[game_id] = {"home_score": home_score, "away_score": away_score}
        return results
    except Exception as e:
        print(f"[Tracker] Error fetching actuals for {date}: {e}")
        return {}
