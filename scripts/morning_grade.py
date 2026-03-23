#!/usr/bin/env python3
"""
morning_grade.py — Run this every morning to grade yesterday's predictions.

What it does:
  1. Grades yesterday's (and today's if needed) predictions against ESPN final scores
  2. Updates the self-learning calibration offsets
  3. Exports the full Excel report (data/prediction_report.xlsx)

Set it up as a daily Mac cron job — see instructions at the bottom of this file.
"""

import sys
import os
from datetime import date, timedelta
from pathlib import Path

# Make sure the project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.prediction_tracker import grade_predictions, get_model_record


def main():
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today     = date.today().strftime("%Y-%m-%d")

    print(f"\n{'='*50}")
    print(f"  NBA Model — Morning Grade  ({today})")
    print(f"{'='*50}\n")

    for day in (yesterday, today):
        try:
            result = grade_predictions(day)
            if result:
                print(f"  {day}:")
                print(f"    Moneyline : {result['moneyline']['correct']}-{result['moneyline']['total'] - result['moneyline']['correct']}  ({result['moneyline']['pct']}%)")
                print(f"    Spread    : {result['spread']['correct']}-{result['spread']['total'] - result['spread']['correct']}  ({result['spread']['pct']}%)")
                print(f"    Over/Under: {result['total']['correct']}-{result['total']['total'] - result['total']['correct']}  ({result['total']['pct']}%)")
        except Exception as e:
            print(f"  {day}: skipped — {e}")

    print()
    record = get_model_record()
    overall = record.get("overall", {})
    print(f"  All-time record: {overall.get('correct', 0)}-{overall.get('total', 0) - overall.get('correct', 0)}  ({overall.get('pct', 0.0)}% win rate)")
    print(f"\n  Excel report updated: data/prediction_report.xlsx")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO SET UP AS A DAILY MAC CRON JOB (runs automatically every morning)
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Open Terminal and run:
#      crontab -e
#
# 2. This opens a text editor. Add this line (runs at 9 AM every day):
#      0 9 * * * /usr/bin/python3 /Users/rodgersbahati/Developer/Projects/NBA/NBA_Player_Stats_Predictor/scripts/morning_grade.py >> /Users/rodgersbahati/Developer/Projects/NBA/NBA_Player_Stats_Predictor/data/grade_log.txt 2>&1
#
# 3. Save and exit (press Escape, then type :wq, then Enter)
#
# 4. Verify it was saved:
#      crontab -l
#
# That's it. Every morning at 9 AM the script runs, grades the previous night's
# games, updates the calibration, and refreshes the Excel report automatically.
# ─────────────────────────────────────────────────────────────────────────────
