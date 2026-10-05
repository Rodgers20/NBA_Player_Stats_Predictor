"""Actual user-entered wagers. No automatic model picks or assumed stakes."""
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from uuid import uuid4
from utils.odds_budget import connection
from utils.market_evaluation import valid_price


def _money(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError('Amount must be finite')
        return int((number * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    except (InvalidOperation, TypeError):
        raise ValueError('Enter a valid amount') from None


def _setup(db):
    db.execute('''CREATE TABLE IF NOT EXISTS personal_bets (
        id TEXT PRIMARY KEY, token TEXT UNIQUE NOT NULL, payload TEXT NOT NULL,
        stake_cents INTEGER NOT NULL, result TEXT NOT NULL, profit_cents INTEGER,
        created_at TEXT NOT NULL, settled_at TEXT)''')


def add_bet(payload, token):
    if not token:
        raise ValueError('Missing submission identifier')
    data = {k: payload.get(k) for k in ('mode','league','player','game_date','stat','side','line','price','book','notes')}
    for key in ('player','book'):
        data[key] = str(data[key] or '').strip()
        if not data[key]:
            raise ValueError(f'{key.title()} is required')
    if data['mode'] not in ('paper','real') or data['league'] not in ('nba','wnba'):
        raise ValueError('Choose paper/real and a league')
    if data['stat'] not in ('PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK',
                            'PTS+REB', 'PTS+AST', 'REB+AST', 'PTS+REB+AST', 'STL+BLK') or data['side'] not in ('Over','Under'):
        raise ValueError('Choose a supported player stat and a side')
    date.fromisoformat(str(data['game_date']))
    if not valid_price(data['price']):
        raise ValueError('Enter American odds, such as -110 or +120')
    data['price'] = float(data['price'])
    try:
        line = Decimal(str(data['line']))
        if not line.is_finite() or line < 0 or line * 2 != (line * 2).to_integral_value():
            raise ValueError()
    except (InvalidOperation, ValueError):
        raise ValueError('Line must be a nonnegative whole or half point') from None
    data['line'] = float(line)
    stake = _money(payload.get('stake'))
    if stake <= 0:
        raise ValueError('Stake must be greater than zero')
    with connection() as db:
        _setup(db)
        identifier = str(uuid4())
        db.execute('INSERT OR IGNORE INTO personal_bets VALUES (?,?,?,?,?,?,?,?)',
                   (identifier, token, json.dumps(data), stake, 'pending', None, datetime.now(timezone.utc).isoformat(), None))
        return db.execute('SELECT id FROM personal_bets WHERE token=?', (token,)).fetchone()[0]


def settle_bet(identifier, result):
    if result not in ('win','loss','push','void','pending'):
        raise ValueError('Invalid result')
    with connection() as db:
        _setup(db)
        row = db.execute('SELECT payload,stake_cents FROM personal_bets WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise ValueError('Choose a recorded bet')
        data, stake = json.loads(row[0]), row[1]
        price = Decimal(str(data['price']))
        payout = price / 100 if price > 0 else 100 / abs(price)
        profit = int((Decimal(stake) * payout).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) if result == 'win' else (-stake if result == 'loss' else 0)
        db.execute('UPDATE personal_bets SET result=?,profit_cents=?,settled_at=? WHERE id=?',
                   (result, None if result == 'pending' else profit,
                    None if result == 'pending' else datetime.now(timezone.utc).isoformat(), identifier))


def list_bets(mode=None):
    with connection() as db:
        _setup(db)
        rows = db.execute('SELECT id,payload,stake_cents,result,profit_cents,created_at FROM personal_bets ORDER BY created_at DESC').fetchall()
    bets = [dict(json.loads(payload), id=identifier, stake_cents=stake, result=result, profit_cents=profit, created_at=at)
            for identifier, payload, stake, result, profit, at in rows]
    return [bet for bet in bets if mode is None or bet['mode'] == mode]


def summary(mode):
    bets = list_bets(mode)
    settled = [b for b in bets if b['result'] in ('win','loss','push')]
    stake = sum(b['stake_cents'] for b in settled)
    profit = sum(b['profit_cents'] for b in settled)
    return {'count':len(bets), 'settled':len(settled), 'profit':profit/100,
            'roi':100*profit/stake if stake else None,
            'pending':sum(b['stake_cents'] for b in bets if b['result']=='pending')/100,
            'staked':stake/100}
