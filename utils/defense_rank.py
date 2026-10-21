"""Defense-vs-position rankings for the props board and the standalone defense page.

Reads the CSVs produced by utils.data_fetch.calculate_defense_vs_position (NBA) and
utils.wnba_loader.load_defense_vs_position (WNBA). The two writers rank in opposite
directions, so ranks are recomputed here from the allowed averages: rank 1 means the
team ALLOWS THE MOST to that position (softest matchup), rank N the least.
"""
from functools import lru_cache
import pandas as pd
from utils.league_config import get_config

STATS = ('PTS', 'REB', 'AST')
_SOURCE_COLUMNS = {'nba': {s: f'AVG_{s}_ALLOWED' for s in STATS}, 'wnba': {s: s for s in STATS}}
# Share of teams (softest first) that earn each grade; the rest get 'F'.
GRADE_BANDS = ((0.2, 'A'), (0.4, 'B'), (0.6, 'C'), (0.8, 'D'))
NO_MATCHUP = dict(def_rank=None, def_rank_of=None, matchup_grade=None)


def grade(rank, of):
    """A (softest) to F (toughest) from a rank among `of` teams."""
    fraction = (rank - 1) / (of - 1) if of > 1 else 0.5
    return next((letter for limit, letter in GRADE_BANDS if fraction <= limit), 'F')


@lru_cache(maxsize=4)
def _load(league, defense_mtime, positions_mtime):
    data_dir = get_config(league).data_dir
    raw = pd.read_csv(data_dir / 'defense_vs_position.csv')
    raw['SEASON'] = raw['SEASON'].astype(str)
    latest = raw[raw['SEASON'] == raw['SEASON'].max()]
    columns = _SOURCE_COLUMNS[league]
    long = pd.concat([
        latest[['TEAM_ABBREVIATION', 'POSITION']].assign(stat=stat, allowed=latest[column].astype(float))
        for stat, column in columns.items()], ignore_index=True)
    long['rank'] = long.groupby(['POSITION', 'stat'])['allowed'].rank(ascending=False, method='min').astype(int)
    long['of'] = long.groupby(['POSITION', 'stat'])['allowed'].transform('count').astype(int)
    positions = pd.read_csv(data_dir / 'player_positions.csv')
    positions['SEASON'] = positions['SEASON'].astype(str)
    positions = positions.sort_values('SEASON').drop_duplicates('PLAYER_NAME', keep='last')
    return str(latest['SEASON'].iloc[0]) if len(latest) else None, long, dict(zip(positions['PLAYER_NAME'], positions['POSITION']))


def _tables(league):
    directory = get_config(league).data_dir
    files = [directory / 'defense_vs_position.csv', directory / 'player_positions.csv']
    if not all(path.exists() for path in files):
        return None, pd.DataFrame(), {}
    return _load(league, *(path.stat().st_mtime_ns for path in files))


def rankings(league, stat=None):
    """{season, rankings: [{team, position, stat, allowed, rank, of}]}; empty when no data."""
    try:
        season, long, _ = _tables(league)
    except (OSError, ValueError, KeyError):
        season, long = None, pd.DataFrame()
    if stat:
        long = long[long['stat'] == stat.upper()] if len(long) else long
    rows = [dict(team=r.TEAM_ABBREVIATION, position=r.POSITION, stat=r.stat, allowed=round(r.allowed, 1),
                 rank=int(r.rank), of=int(r.of))
            for r in long.sort_values(['stat', 'POSITION', 'rank']).itertuples()] if len(long) else []
    return dict(season=season, rankings=rows)


def matchup(league, player, opponent, stat):
    """{def_rank, def_rank_of, matchup_grade} for a player's position vs `opponent`, or all None."""
    if stat not in STATS or not opponent:
        return dict(NO_MATCHUP)
    try:
        _, long, positions = _tables(league)
    except (OSError, ValueError, KeyError):
        return dict(NO_MATCHUP)
    position = positions.get(player)
    if position is None or long.empty:
        return dict(NO_MATCHUP)
    hit = long[(long['stat'] == stat) & (long['POSITION'] == position) & (long['TEAM_ABBREVIATION'] == opponent)]
    if hit.empty:
        return dict(NO_MATCHUP)
    rank, of = int(hit['rank'].iloc[0]), int(hit['of'].iloc[0])
    return dict(def_rank=rank, def_rank_of=of, matchup_grade=grade(rank, of))
