"""Research fields layered onto evaluated prop rows: line shopping, hit windows, teammate splits.

Each helper is pure or reads only local data, never buys odds, and degrades to empty values.
"""
from utils import defense_rank, line_shopping
from utils.hit_rate_windows import h2h_hit_rate, hit_rate, hit_rate_windows
from utils.line_snapshots import books_of
from utils.teammate_impact import analyze_teammate_impact

# Defaults for rows that carry no research data (research shortlist, WNBA fallback, cached NBA rows).
EXTRA_DEFAULTS = dict(books=[], best_over=None, best_under=None, consensus=None, open_line=None,
                      hit_windows=None, teammate_splits=[], def_rank_of=None, matchup_grade=None)
KEY_TEAMMATE_MINUTES = 20      # average minutes over the last 10 games
MAX_TEAMMATES = 3
UNAVAILABLE = ('OUT', 'DOUBTFUL', 'QUESTIONABLE')


def quote_fields(quote, opens, league, player, stat):
    """books / best_over / best_under / consensus / open_line from a stored quote."""
    books = books_of(quote)
    return dict(books=books, **line_shopping.summarize_books(books),
                open_line=opens.get((player, stat, quote.get('bookmaker'))))


def hit_window_fields(player_history, stat, line, direction, opponent, target):
    """L5/L10/L20/season windows plus head-to-head, from games before `target` (history is newest first)."""
    past = player_history[player_history['_date'] < target]
    if past.empty or stat not in past:
        return None
    season = past[past['SEASON'] == past['SEASON'].iloc[0]] if 'SEASON' in past else past
    windows = hit_rate_windows(list(past[stat]), line, direction, list(season[stat]))
    versus = past[past['MATCHUP'].astype(str).str.split().str[-1] == opponent] if opponent and 'MATCHUP' in past else past.iloc[0:0]
    return dict(windows, h2h=h2h_hit_rate(list(versus[stat]), line, direction))


def unavailable_teammates(league, team, injuries_for):
    try:
        return [i for i in injuries_for(league, team) if i.get('status') in UNAVAILABLE]
    except Exception as exc:  # injury feeds are best-effort; a failure only hides the splits
        print(f'[props] injuries unavailable for {team}: {exc}')
        return []


def teammate_impacts(player, stat, player_history, history, injured):
    """analyze_teammate_impact for each key unavailable teammate; computed once per player/stat."""
    impacts = []
    for entry in injured:
        name = entry.get('name')
        if not name or name == player:
            continue
        games = history[history['PLAYER_NAME'] == name]
        recent_minutes = games.sort_values('_date', ascending=False)['MIN'].head(10).mean() if len(games) else 0
        if not recent_minutes >= KEY_TEAMMATE_MINUTES:
            continue
        impact = analyze_teammate_impact(player, name, player_history, games, stat)
        if impact:
            impacts.append((recent_minutes, dict(impact, status=entry['status'])))
    impacts.sort(key=lambda pair: pair[0], reverse=True)
    return [impact for _, impact in impacts[:MAX_TEAMMATES]]


def teammate_split_rows(impacts, line, direction):
    return [dict(teammate=i['teammate'], status=i['status'], avg_with=i['avg_with'],
                 avg_without=i['avg_without'], games_without=i['games_without'],
                 hit_rate_without=hit_rate(i['values_without'], line, direction)['pct'])
            for i in impacts]


def shared_fields(league, player, stat, quote, opponent, opens):
    """Fields identical for Over and Under: line shopping and defense-vs-position."""
    return dict(quote_fields(quote, opens, league, player, stat),
                **defense_rank.matchup(league, player, opponent, stat))
