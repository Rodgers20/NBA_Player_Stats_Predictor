"""Eligibility for priced recommendations, independent of analysis projections."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import math
import pandas as pd

ET = ZoneInfo('America/New_York')


def quote_problem(quote, target_date=None):
    if not quote or not quote.get('event_id'):
        return 'No verified event for this quote'
    now = datetime.now(timezone.utc)
    try:
        start = datetime.fromisoformat(quote['commence_time'].replace('Z', '+00:00'))
        if start.tzinfo is None or start <= now:
            return 'Game already started or start time unavailable'
        expected = str(target_date or now.astimezone(ET).date())[:10]
        if start.astimezone(ET).date().isoformat() != expected:
            return 'Quote belongs to a different game date'
        fetched = float(quote['fetched_at'])
        updated = datetime.fromisoformat(quote['updated_at'].replace('Z', '+00:00'))
        if updated.tzinfo is None:
            return 'Quote update time has no timezone'
        for age in (now.timestamp() - fetched, (now-updated).total_seconds()):
            if not math.isfinite(age) or age < -60 or age >= 1800:
                return 'Quote expired; refresh or enter the current book price'
    except (KeyError, ValueError, TypeError, AttributeError):
        return 'Quote timestamp unavailable'
    return None


def history_problem(history, target_date=None):
    if history.empty:
        return 'No player history'
    date_column = 'GAME_DATE' if 'GAME_DATE' in history else '_date'
    days = pd.to_datetime(history[date_column], format='mixed', errors='coerce')
    when = pd.Timestamp(target_date or datetime.now(ET).date()).normalize()
    completed = days[days < when]
    if completed.empty or len(completed.drop_duplicates()) < 10:
        return 'Fewer than 10 completed appearances'
    if (when - completed.max()).days > 14:
        return 'Player history is older than 14 days; refresh data before evaluating bets'
    return None


def history_label(history):
    if history.empty:
        return 'No player history available'
    key = 'GAME_DATE' if 'GAME_DATE' in history else '_date'
    last = pd.to_datetime(history[key], format='mixed', errors='coerce').max()
    if pd.isna(last):
        return 'History date unavailable'
    age = (pd.Timestamp(datetime.now(ET).date()) - last.normalize()).days
    return f'History through {last.date()} · {age} days old' + (' · Stale for bet evaluation' if age > 14 else '')
