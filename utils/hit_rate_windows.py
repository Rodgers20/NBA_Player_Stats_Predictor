"""Hit rates over recent windows (pure).

A game that lands exactly on the line is a push: it is neither a hit nor a miss and
is excluded from `total`, so pct = hits / (hits + misses).
"""
WINDOWS = {'L5': 5, 'L10': 10, 'L20': 20}


def _rate(values, line, direction):
    under = str(direction).lower() == 'under'
    graded = [v for v in values if v is not None and v != line]
    hits = sum((v < line) if under else (v > line) for v in graded)
    total = len(graded)
    return dict(hits=hits, total=total, pct=round(100 * hits / total, 1) if total else None)


def hit_rate(values, line, direction):
    """{hits, total, pct} for any list of stat values."""
    return _rate(list(values), line, direction)


def hit_rate_windows(values_desc, line, direction, season_values_desc=None):
    """L5/L10/L20/season hit rates; `values_desc` is most-recent-first.

    `season` uses `season_values_desc` when given, else all `values_desc`.
    """
    values = list(values_desc)
    windows = {name: _rate(values[:size], line, direction) for name, size in WINDOWS.items()}
    season = values if season_values_desc is None else list(season_values_desc)
    return dict(windows, season=_rate(season, line, direction))


def h2h_hit_rate(opponent_values_desc, line, direction):
    """Hit rate over games against today's opponent (caller pre-filters by opponent)."""
    return _rate(list(opponent_values_desc), line, direction)
