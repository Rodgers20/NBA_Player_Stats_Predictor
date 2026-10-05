"""Persistent, conservative shared free-tier budget. Never purchase extra credits."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parents[1] / 'data' / 'personal.sqlite3'
DAILY_LIMIT = 12
MONTHLY_LIMIT = 400


@contextmanager
def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=10) as db:
        db.execute('CREATE TABLE IF NOT EXISTS odds_calls (at TEXT NOT NULL, cost INTEGER NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS odds_status (id INTEGER PRIMARY KEY, remaining INTEGER, message TEXT, at TEXT)')
        yield db


def status():
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        daily = db.execute('SELECT COALESCE(SUM(cost),0) FROM odds_calls WHERE substr(at,1,10)=?', (now[:10],)).fetchone()[0]
        monthly = db.execute('SELECT COALESCE(SUM(cost),0) FROM odds_calls WHERE substr(at,1,7)=?', (now[:7],)).fetchone()[0]
        row = db.execute('SELECT remaining,message,at FROM odds_status WHERE id=1').fetchone()
    return dict(daily=daily, monthly=monthly, daily_limit=DAILY_LIMIT, monthly_limit=MONTHLY_LIMIT,
                remaining=row[0] if row else None, message=row[1] if row else 'No odds requests yet',
                checked_at=row[2] if row else None)


def report(message, remaining=None):
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        previous = db.execute('SELECT remaining FROM odds_status WHERE id=1').fetchone()
        if remaining is None and previous:
            remaining = previous[0]
        db.execute('INSERT OR REPLACE INTO odds_status VALUES (1,?,?,?)', (remaining, message, now))


def reserve(cost):
    """Reserve worst-case credits atomically, including failed requests."""
    if cost <= 0:
        return True
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        db.execute('BEGIN IMMEDIATE')
        daily = db.execute('SELECT COALESCE(SUM(cost),0) FROM odds_calls WHERE substr(at,1,10)=?', (now[:10],)).fetchone()[0]
        monthly = db.execute('SELECT COALESCE(SUM(cost),0) FROM odds_calls WHERE substr(at,1,7)=?', (now[:7],)).fetchone()[0]
        row = db.execute('SELECT remaining FROM odds_status WHERE id=1').fetchone()
        remaining = row[0] if row else None
        if daily + cost > DAILY_LIMIT or monthly + cost > MONTHLY_LIMIT or (remaining is not None and remaining < cost):
            return False
        db.execute('INSERT INTO odds_calls VALUES (?,?)', (now, cost))
        if remaining is not None:
            db.execute('UPDATE odds_status SET remaining=? WHERE id=1', (max(0, remaining-cost),))
    return True


def request(get, url, params, timeout=10):
    """Wrap requests.get without ever logging a URL containing the API key."""
    cost = len(params.get('markets', '').split(',')) if '/odds' in url else 0
    if not reserve(cost):
        report('Free credit limit reached. Use a manual line or wait for the budget to reset.')
        return None
    try:
        response = get(url, params=params, timeout=timeout)
    except Exception:
        report('Odds provider unreachable; no new quotes available.')
        return None
    if response is None:
        return None
    raw = response.headers.get('x-requests-remaining')
    remaining = int(raw) if raw is not None and str(raw).isdigit() else None
    if response.status_code != 200:
        report(f'Odds provider HTTP {response.status_code}; check account/quota. No new quotes.', remaining)
        return None
    report('Odds provider connected.', remaining)
    return response
