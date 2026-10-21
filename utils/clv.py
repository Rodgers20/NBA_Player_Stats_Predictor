"""Closing-line value (CLV), pure.

CLV compares the price a bet got with the market's final price on the same side.
Positive means the bettor beat the close. It is independent of the bet result, so a
pushed bet still has CLV.

Price CLV is the gain in implied probability (percentage points) from the bet price
to the close price. When the line itself moved we add a rough, documented
approximation: LINE_HALF_POINT_PCT percentage points per half point moved in the
bettor's favour (a lower line for Overs, a higher line for Unders).
"""
from utils.market_evaluation import valid_price

LINE_HALF_POINT_PCT = 2.0  # approximation; real value depends on stat and scoring variance
EMPTY = {'close_line': None, 'close_price': None, 'clv_pct': None}


def implied_probability(price):
    price = float(price)
    return 100 / (price + 100) if price > 0 else abs(price) / (abs(price) + 100)


def _side_keys(side):
    if str(side).lower() not in ('over', 'under'):
        raise ValueError("side must be 'Over' or 'Under'")
    return str(side).lower() + '_price'


def closing_quote(snapshots, cutoff_ts, book=None):
    """Latest snapshot per book at or before `cutoff_ts`; the bet's book if given, else consensus."""
    from utils.line_shopping import consensus
    latest = {}
    for snap in sorted(snapshots, key=lambda s: s['ts']):
        if snap['ts'] <= cutoff_ts:
            latest[snap['book']] = snap
    if not latest:
        return None
    if book and book in latest:
        return latest[book]
    agreed = consensus(list(latest.values()))
    return agreed if agreed and agreed['line'] is not None else None


def compute_clv(side, line, price, close_line, close_price):
    """Return {close_line, close_price, clv_pct} for a bet; all None without a usable close."""
    if close_line is None or not valid_price(close_price) or not valid_price(price):
        return dict(EMPTY)
    over = _side_keys(side) == 'over_price'
    moved = (float(close_line) - float(line)) if over else (float(line) - float(close_line))
    price_gain = implied_probability(close_price) - implied_probability(price)
    clv = 100 * price_gain + (moved / 0.5) * LINE_HALF_POINT_PCT
    return dict(close_line=float(close_line), close_price=float(close_price), clv_pct=round(clv, 2))


def clv_from_snapshots(side, line, price, snapshots, cutoff_ts, book=None):
    """Convenience: pick the closing quote from snapshots, then compute CLV for the side."""
    close = closing_quote(snapshots, cutoff_ts, book)
    if close is None:
        return dict(EMPTY)
    return compute_clv(side, line, price, close['line'], close.get(_side_keys(side)))
