"""Pure line-shopping summaries over the per-book quotes kept by the odds parsers."""
from collections import Counter
from statistics import median
from utils.market_evaluation import valid_price


def _best(books, price_key, lower_line_better):
    """Best book for one side. The line dominates; American odds order numerically by payout."""
    candidates = [b for b in books if valid_price(b.get(price_key)) and b.get('line') is not None]
    if not candidates:
        return None
    sign = -1 if lower_line_better else 1
    best = max(candidates, key=lambda b: (sign * b['line'], float(b[price_key])))
    return {'book': best['book'], 'price': best[price_key], 'line': best['line']}


def consensus(books):
    """Modal line (ties go to the lower line); prices are medians of the books quoting that line."""
    lines = [b['line'] for b in books if b.get('line') is not None]
    if not lines:
        return None
    counts = Counter(lines)
    line = min(counts, key=lambda value: (-counts[value], value))
    at_line = [b for b in books if b.get('line') == line]

    def mid(key):
        prices = [float(b[key]) for b in at_line if valid_price(b.get(key))]
        return int(round(median(prices))) if prices else None

    return {'line': line, 'over_price': mid('over_price'), 'under_price': mid('under_price')}


def summarize_books(books):
    books = list(books or [])
    return {'best_over': _best(books, 'over_price', True),
            'best_under': _best(books, 'under_price', False),
            'consensus': consensus(books)}
