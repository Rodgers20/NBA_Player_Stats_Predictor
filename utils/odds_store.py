"""Disk-backed quote store so a restart or cache expiry never discards paid-for prices."""
import json
import time
from utils import odds_budget

KEEP_SECONDS = 2 * 24 * 3600


def save(league, quotes):
    """Upsert {player: {stat: quote}}; newer quotes replace older ones per player/stat."""
    now = time.time()
    rows = [(league, player, stat, json.dumps(quote), now)
            for player, markets in quotes.items() for stat, quote in markets.items()]
    with odds_budget.connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS odds_quotes (league TEXT, player TEXT, stat TEXT, payload TEXT, saved REAL, '
                   'PRIMARY KEY (league, player, stat))')
        db.executemany('INSERT OR REPLACE INTO odds_quotes VALUES (?,?,?,?,?)', rows)
        db.execute('DELETE FROM odds_quotes WHERE saved < ?', (now - KEEP_SECONDS,))


def load(league):
    with odds_budget.connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS odds_quotes (league TEXT, player TEXT, stat TEXT, payload TEXT, saved REAL, '
                   'PRIMARY KEY (league, player, stat))')
        rows = db.execute('SELECT player, stat, payload FROM odds_quotes WHERE league=?', (league,)).fetchall()
    out = {}
    for player, stat, payload in rows:
        out.setdefault(player, {})[stat] = json.loads(payload)
    return out
