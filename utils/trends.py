"""Plain-language trend cards from player game logs (pure; no I/O, no odds fetches).

A card is "Under 5.5 rebounds in 9 of last 10". The line is the live sportsbook line when a
quote is supplied for that player/stat, otherwise the most decisive non-trivial half-point line
the last ten games support (a derived line, labelled as such).
"""
import math
import pandas as pd

WINDOW = 10
MIN_HITS = 8
STATS = ('PTS', 'REB', 'AST', 'FG3M')
NOUNS = {'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'threes'}
MIN_AVG = {'PTS': 8, 'REB': 3, 'AST': 2, 'FG3M': 1}   # skip players whose stat is negligible
MIN_MINUTES = 15
RECENT_DAYS = 14
# ponytail: derived lines stay within 60%-140% of the average so "Over 0.5" never headlines; tune if too loose.
OVER_FLOOR, UNDER_CEILING = 0.6, 1.4
STARS_BY_HITS = {10: 5, 9: 4, 8: 3}


def _hits(values, line, side):
    return [(v > line) if side == 'over' else (v < line) for v in values]


def _derived_line(values, avg, side):
    """Line with the most hits (ties: most demanding) inside the non-trivial band, or None."""
    candidates = {x + d for x in values for d in (-0.5, 0.5)}
    if side == 'over':
        band = [line for line in candidates if line >= avg * OVER_FLOOR]
        pick = lambda line: (sum(_hits(values, line, side)), line)
    else:
        band = [line for line in candidates if line <= avg * UNDER_CEILING]
        pick = lambda line: (sum(_hits(values, line, side)), -line)
    return max(band, key=pick) if band else None


def _card(player, team, opponent, stat, side, line, values, source, live):
    results = _hits(values, line, side)
    hits = sum(results)
    if hits < MIN_HITS:
        return None
    return dict(player=player, team=team, opponent=opponent, stat=stat, side=side, line=line,
                hits=hits, games=len(values), results=results, avg=round(sum(values) / len(values), 1),
                text=f'{side.capitalize()} {line:g} {NOUNS[stat]} in {hits} of last {len(values)}',
                stars=STARS_BY_HITS.get(hits, 3), line_source=source, live=live)


def _live(quote, side):
    return dict(line=quote['line'], price=quote.get(f'{side}_price'), book=quote.get('bookmaker'))


def _opponent(quote, team):
    if not quote:
        return None
    return quote.get('away_team') if quote.get('home_team') == team else quote.get('home_team') if quote.get('away_team') == team else None


def _player_cards(name, rows, quotes):
    team = str(rows['TEAM_ABBREVIATION'].iloc[-1]) if 'TEAM_ABBREVIATION' in rows else ''
    cards = []
    for stat in STATS:
        values = pd.to_numeric(rows[stat], errors='coerce').tolist()
        if not all(math.isfinite(v) for v in values):
            continue
        avg = sum(values) / len(values)
        if avg < MIN_AVG[stat]:
            continue
        quote = (quotes.get(name) or {}).get(stat)
        if quote and quote.get('line') is not None:
            line = float(quote['line'])
            cards += [_card(name, team, _opponent(quote, team), stat, side, line, values, 'live', _live(quote, side))
                      for side in ('over', 'under')]
        else:
            for side in ('over', 'under'):
                line = _derived_line(values, avg, side)
                if line is not None:
                    cards.append(_card(name, team, None, stat, side, line, values, 'derived', None))
    return [c for c in cards if c]


def build_trends(history, quotes=None, today=None, recent_days=RECENT_DAYS):
    """Cards for players with a full window of recent games, strongest first."""
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    past = history[history['_date'] < today].sort_values('_date')
    cards = []
    for name, games in past.groupby('PLAYER_NAME', sort=False):
        rows = games.tail(WINDOW)
        if len(rows) < WINDOW or (today - rows['_date'].iloc[-1]).days > recent_days:
            continue
        if 'MIN' in rows and pd.to_numeric(rows['MIN'], errors='coerce').mean() < MIN_MINUTES:
            continue
        cards += _player_cards(name, rows, quotes or {})
    return sorted(cards, key=lambda c: (-c['stars'], -c['hits'], c['player'], c['stat']))
