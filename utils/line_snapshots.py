"""Append-only per-book line history in SQLite, written whenever quotes are saved.

Identical consecutive rows (same book/line/prices as the latest stored row) are skipped
so repeated refreshes do not bloat the table. Rows older than KEEP_DAYS are pruned.
"""
import time
from utils import odds_budget

KEEP_DAYS = 30
_SCHEMA = ('CREATE TABLE IF NOT EXISTS line_snapshots (league TEXT, player TEXT, stat TEXT, book TEXT, '
           'line REAL, over_price REAL, under_price REAL, ts REAL)')
_INDEX = 'CREATE INDEX IF NOT EXISTS line_snapshots_key ON line_snapshots (league, player, stat, book, ts)'
_FIELDS = ('book', 'line', 'over_price', 'under_price')


def books_of(quote):
    """Per-book quotes of a stored quote; legacy single-book quotes become a one-item list."""
    books = quote.get('books')
    if books:
        return list(books)
    if quote.get('line') is None:
        return []
    return [dict(book=quote.get('bookmaker') or 'Unknown', line=quote['line'],
                 over_price=quote.get('over_price'), under_price=quote.get('under_price'))]


def record(db, league, quotes, now=None):
    """Append changed snapshots for {player: {stat: quote}} using an open connection."""
    now = time.time() if now is None else now
    db.execute(_SCHEMA)
    db.execute(_INDEX)
    for player, markets in quotes.items():
        for stat, quote in markets.items():
            for entry in books_of(quote):
                values = tuple(entry.get(key) for key in _FIELDS)
                if values[0] is None or values[1] is None:
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
                          'WHERE league=? AND player=? AND stat=? ORDER BY ts ASC',
                          (league, player, stat)).fetchall()
    return [dict(book=b, line=l, over_price=o, under_price=u, ts=t) for b, l, o, u, t in rows]


def open_lines(league):
    """{(player, stat, book): earliest stored line} for opening-line display."""
    with odds_budget.connection() as db:
        db.execute(_SCHEMA)
        rows = db.execute('SELECT player, stat, book, line FROM line_snapshots WHERE league=? ORDER BY ts DESC',
                          (league,)).fetchall()
    return {(player, stat, book): line for player, stat, book, line in rows}  # last write = earliest


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
