"""Append-only per-book line history in SQLite, written whenever quotes are saved.

Identical consecutive rows (same book/line/prices as the latest stored row) are skipped
so repeated refreshes do not bloat the table. Rows older than KEEP_DAYS are pruned.
"""
import math
import time
from utils import odds_budget

KEEP_DAYS = 30
HISTORY_LIMIT = 2000
_SCHEMA = ('CREATE TABLE IF NOT EXISTS line_snapshots (league TEXT, player TEXT, stat TEXT, book TEXT, '
           'line REAL, over_price REAL, under_price REAL, ts REAL)')
_INDEX = 'CREATE INDEX IF NOT EXISTS line_snapshots_key ON line_snapshots (league, player, stat, book, ts)'


def books_of(quote):
    """Per-book quotes of a stored quote; legacy single-book quotes become a one-item list."""
    books = quote.get('books')
    if isinstance(books, list) and books:
        return list(books)
    if quote.get('line') is None:
        return []
    return [dict(book=quote.get('bookmaker') or 'Unknown', line=quote['line'],
                 over_price=quote.get('over_price'), under_price=quote.get('under_price'))]


def _number(value):
    """Finite float, or None for anything non-numeric (bool, NaN, inf, str)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _snapshot_values(entry):
    """(book, line, over, under) for a well-formed book entry, else None."""
    if not isinstance(entry, dict) or not isinstance(entry.get('book'), str):
        return None
    line = _number(entry.get('line'))
    if line is None:
        return None
    return entry['book'], line, _number(entry.get('over_price')), _number(entry.get('under_price'))


def record(db, league, quotes, now=None):
    """Append changed snapshots for {player: {stat: quote}} using an open connection."""
    now = time.time() if now is None else now
    db.execute(_SCHEMA)
    db.execute(_INDEX)
    for player, markets in quotes.items():
        for stat, quote in markets.items():
            for entry in books_of(quote):
                values = _snapshot_values(entry)
                if values is None:
                    continue
                last = db.execute('SELECT line, over_price, under_price FROM line_snapshots '
                                  'WHERE league=? AND player=? AND stat=? AND book=? ORDER BY ts DESC LIMIT 1',
                                  (league, player, stat, values[0])).fetchone()
                if last == values[1:]:
                    continue
                db.execute('INSERT INTO line_snapshots VALUES (?,?,?,?,?,?,?,?)',
                           (league, player, stat, *values, now))
    db.execute('DELETE FROM line_snapshots WHERE ts < ?', (now - KEEP_DAYS * 86400,))


def history(league, player, stat):
    """Snapshots for one market, ascending by time."""
    with odds_budget.connection() as db:
        db.execute(_SCHEMA)
        rows = db.execute('SELECT book, line, over_price, under_price, ts FROM line_snapshots '
                          'WHERE league=? AND player=? AND stat=? ORDER BY ts DESC LIMIT ?',
                          (league, player, stat, HISTORY_LIMIT)).fetchall()
    return [dict(book=b, line=l, over_price=o, under_price=u, ts=t) for b, l, o, u, t in reversed(rows)]


def open_lines(league):
    """{(player, stat, book): earliest stored line} for opening-line display."""
    with odds_budget.connection() as db:
        db.execute(_SCHEMA)
        rows = db.execute('SELECT player, stat, book, line, MIN(ts) FROM line_snapshots WHERE league=? '
                          'GROUP BY player, stat, book', (league,)).fetchall()
    return {(player, stat, book): line for player, stat, book, line, _ in rows}  # bare column = row of MIN(ts)


def day_cutoff_ts(game_date):
    """End of `game_date` (UTC) as a timestamp. Tip time is not stored, so the last
    snapshot of the game day stands in for the close (pregame props vanish at tip)."""
    from datetime import date, datetime, time as dtime, timedelta, timezone
    day = date.fromisoformat(str(game_date)) + timedelta(days=1)
    return datetime.combine(day, dtime.min, tzinfo=timezone.utc).timestamp()


def clv_for_bet(league, player, stat, side, line, price, game_date, book=None):
    """CLV for a bet from stored snapshots; all-None when no snapshot exists before the cutoff."""
    from utils import clv
    snaps = history(league, player, stat)
    return clv.clv_from_snapshots(side, line, price, snaps, day_cutoff_ts(game_date), book)
