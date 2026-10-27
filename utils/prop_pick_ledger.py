"""Published prop picks, kept at their first price and graded against the box score and the closing line.

A pick enters the ledger the first time the board publishes it with a sportsbook price, before its game
tips; later refreshes never move its line or price. The morning after, each pick gets its result, its
profit in units at the published price, and its closing-line value from stored line snapshots. Closing-line
value (CLV) is the honest early test of a betting model: results take hundreds of bets to separate skill
from luck, while beating the close shows up much sooner.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import tempfile
from zoneinfo import ZoneInfo

import pandas as pd

VOID_AFTER_DAYS = 3   # no box-score row this long after the game: the player sat, and books void the bet
KEEP = ('player', 'team', 'opponent', 'stat', 'direction', 'line', 'price', 'book', 'ev', 'model_prob',
        'market_prob', 'game_matchup')


def _path(league, path=None):
    if league not in ('nba', 'wnba'):
        raise ValueError('Invalid league')
    return Path(path) if path else Path(__file__).resolve().parents[1] / 'data' / league / 'prop_pick_ledger.json'


def load(league, path=None):
    target = _path(league, path)
    return json.loads(target.read_text()) if target.exists() else {'version': 1, 'league': league, 'picks': []}


def _save(ledger, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=target.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(ledger, f, indent=2, allow_nan=False)
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _tipoff(game):
    try:
        return datetime.fromisoformat(str(game.get('start_time')).replace('Z', '+00:00'))
    except ValueError:
        return None


def _identity(pick):
    return (pick['date'], pick['player'], pick['stat'], pick['direction'])


def record_picks(league, board, games, path=None, now=None):
    """Add the board's priced picks whose games have not started. Returns how many were new."""
    now = now or datetime.now(timezone.utc)
    date = board.get('target_date')
    if not date:
        return 0
    upcoming = {g.get('matchup') for g in (games or {}).get('games', [])
                if g.get('status_state', 'pre') == 'pre' and (_tipoff(g) or now) > now}
    target = _path(league, path)
    ledger = load(league, target)
    seen = {_identity(p) for p in ledger['picks']}
    added = 0
    for prop in board.get('props', []):
        if prop.get('pick_type') != 'pick' or prop.get('price') is None or prop.get('line') is None:
            continue
        if prop.get('game_matchup') not in upcoming:
            continue
        entry = {key: prop.get(key) for key in KEEP}
        entry.update(date=date, published_at=now.isoformat(), actual=None, result=None, units=None,
                     close_line=None, close_price=None, clv_pct=None)
        if _identity(entry) in seen:
            continue
        ledger['picks'].append(entry)
        seen.add(_identity(entry))
        added += 1
    if added:
        _save(ledger, target)
    return added


def _actual(rows, stat):
    """The stat (or combo sum) from one player's box-score row, or None when a part is missing."""
    parts = stat.split('+')
    if rows.empty or not all(part in rows for part in parts):
        return None
    values = pd.to_numeric(rows.iloc[0][parts], errors='coerce')
    return None if values.isna().any() else float(values.sum())


def _settle(direction, line, actual, price):
    over = str(direction).lower() == 'over'
    if actual == line:
        return 'push', 0.0
    won = actual > line if over else actual < line
    if not won:
        return 'loss', -1.0
    return 'win', price / 100 if price > 0 else 100 / abs(price)


def _closing(league, pick):
    from utils import line_snapshots
    return line_snapshots.clv_for_bet(league, pick['player'], pick['stat'], pick['direction'], pick['line'],
                                      pick['price'], pick['date'], pick.get('book'))


def grade_picks(league, history, path=None, today=None, closing=None):
    """Settle every ungraded pick from an earlier date. `history` has PLAYER_NAME, _date and stat columns."""
    # Eastern date: the UTC date turns over at 8 PM ET, while that night's games are still being played.
    today = pd.Timestamp(today or datetime.now(ZoneInfo('America/New_York')).date()).normalize()
    closing = closing or (lambda pick: _closing(league, pick))
    target = _path(league, path)
    ledger = load(league, target)
    graded = voided = 0
    for pick in ledger['picks']:
        day = pd.Timestamp(pick['date'])
        if pick.get('result') or day >= today:
            continue
        rows = history[(history['PLAYER_NAME'] == pick['player']) & (history['_date'].dt.normalize() == day)]
        actual = _actual(rows, pick['stat'])
        if actual is None:
            if (today - day).days > VOID_AFTER_DAYS:
                pick['result'], pick['units'] = 'void', 0.0
                voided += 1
            continue
        pick['actual'] = actual
        pick['result'], pick['units'] = _settle(pick['direction'], float(pick['line']), actual, float(pick['price']))
        try:
            pick.update(closing(pick))
        except Exception as exc:  # a missing snapshot table must not block grading
            print(f'[prop ledger] closing line unavailable for {pick["player"]}: {exc}')
        graded += 1
    if graded or voided:
        _save(ledger, target)
    return graded


def _pct(part, whole):
    return round(100 * part / whole, 1) if whole else 0.0


def summary(ledger):
    """Record for the Props "Record" view. hit/miss/total/pct/by_stat keep the legacy response shape."""
    picks = ledger.get('picks', [])
    settled = [p for p in picks if p.get('result') in ('win', 'loss', 'push')]
    decided = [p for p in settled if p['result'] != 'push']
    wins = sum(p['result'] == 'win' for p in decided)
    with_close = [p['clv_pct'] for p in settled if p.get('clv_pct') is not None]
    by_stat = {}
    for p in decided:
        tally = by_stat.setdefault(p['stat'], {'hit': 0, 'total': 0})
        tally['hit'] += p['result'] == 'win'
        tally['total'] += 1
    recent_dates = sorted({p['date'] for p in decided}, reverse=True)[:7]
    recent = [p for p in decided if p['date'] in recent_dates]
    return {
        'league': ledger.get('league'),
        'hit': wins, 'miss': len(decided) - wins, 'pushes': len(settled) - len(decided), 'total': len(decided),
        'pct': _pct(wins, len(decided)),
        'recent_7d': _pct(sum(p['result'] == 'win' for p in recent), len(recent)),
        'units': round(sum(p['units'] for p in settled), 2),
        'roi_pct': _pct(sum(p['units'] for p in settled), len(settled)),
        'clv': {'n': len(with_close),
                'avg_pct': round(sum(with_close) / len(with_close), 2) if with_close else None,
                'beat_pct': _pct(sum(v > 0 for v in with_close), len(with_close))},
        'pending': sum(p.get('result') is None for p in picks),
        'since': min((p['date'] for p in picks), default=None),
        'by_stat': {stat: {**t, 'pct': _pct(t['hit'], t['total'])} for stat, t in by_stat.items()},
        'recent': sorted(settled, key=lambda p: p['date'], reverse=True)[:40],
    }
