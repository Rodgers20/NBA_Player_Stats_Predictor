"""Actual user-entered wagers. No automatic model picks or assumed stakes."""
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import math
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


def _close_fields(payload):
    """Optional closing line/price entered by the user; only validated keys that were supplied."""
    out = {}
    if payload.get('close_line') is not None:
        try:
            out['close_line'] = float(payload['close_line'])
        except (TypeError, ValueError):
            raise ValueError('Closing line must be a number') from None
        if not math.isfinite(out['close_line']) or out['close_line'] < 0:
            raise ValueError('Closing line must be a nonnegative number')
    if payload.get('close_price') is not None:
        if not valid_price(payload['close_price']):
            raise ValueError('Closing price must be American odds, such as -110 or +120')
        out['close_price'] = float(payload['close_price'])
    return out


def bet_clv(bet):
    """CLV for a journal entry: its own close fields if set, else stored line snapshots.

    Returns {close_line, close_price, clv_pct}; all None when no close is known.
    """
    from utils import clv, line_snapshots
    if bet.get('close_line') is not None and bet.get('close_price') is not None:
        return clv.compute_clv(bet['side'], bet['line'], bet['price'], bet['close_line'], bet['close_price'])
    return line_snapshots.clv_for_bet(bet['league'], bet['player'], bet['stat'], bet['side'],
                                      bet['line'], bet['price'], bet['game_date'], bet.get('book'))


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
    data.update(_close_fields(payload))
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


ODDS_RANGES = ('< -150', '-150 to +100', '+100 to +200', '> +200')  # keep in sync with local-journal.mjs


def odds_range(price):
    if price < -150:
        return ODDS_RANGES[0]
    if price <= 100:
        return ODDS_RANGES[1]
    return ODDS_RANGES[2] if price <= 200 else ODDS_RANGES[3]


def _breakdown(settled, key, fixed_order=None):
    """Record/profit/ROI per group, most bets first (or a fixed order); mirrors local-journal.mjs."""
    groups = {}
    for bet in settled:
        groups.setdefault(key(bet), []).append(bet)
    rows = []
    for name, items in groups.items():
        stake = sum(b['stake_cents'] for b in items)
        profit = sum(b['profit_cents'] for b in items)
        rows.append({'key': name, 'bets': len(items),
                     'wins': sum(b['result'] == 'win' for b in items),
                     'losses': sum(b['result'] == 'loss' for b in items),
                     'pushes': sum(b['result'] == 'push' for b in items),
                     'staked': stake / 100, 'profit': profit / 100,
                     'roi': 100 * profit / stake if stake else None})
    if fixed_order:
        return sorted(rows, key=lambda r: fixed_order.index(r['key']))
    return sorted(rows, key=lambda r: (-r['bets'], r['key']))


def _curve(settled):
    """Cumulative P&L in settlement order (game date, then entry time) and its max drawdown."""
    cents, peak, drawdown, points = 0, 0, 0, []
    for n, bet in enumerate(sorted(settled, key=lambda b: (b['game_date'], b['created_at'])), 1):
        cents += bet['profit_cents']
        peak = max(peak, cents)
        drawdown = max(drawdown, peak - cents)
        points.append({'n': n, 'date': bet['game_date'], 'pnl': cents / 100})
    return points, drawdown / 100


def summary(mode):
    bets = list_bets(mode)
    settled = [b for b in bets if b['result'] in ('win','loss','push')]
    stake = sum(b['stake_cents'] for b in settled)
    profit = sum(b['profit_cents'] for b in settled)
    curve, drawdown = _curve(settled)
    return {'count':len(bets), 'settled':len(settled), 'profit':profit/100,
            'roi':100*profit/stake if stake else None,
            'pending':sum(b['stake_cents'] for b in bets if b['result']=='pending')/100,
            'staked':stake/100,
            'by_stat': _breakdown(settled, lambda b: b['stat']),
            'by_side': _breakdown(settled, lambda b: b['side']),
            'by_book': _breakdown(settled, lambda b: b['book']),
            'by_odds': _breakdown(settled, lambda b: odds_range(b['price']), ODDS_RANGES),
            'curve': curve, 'max_drawdown': drawdown}
