"""First-quarter player lines joined onto full-game history as PTS_1Q, REB_1Q, AST_1Q and FG3M_1Q (pure; no I/O)."""
import pandas as pd

BASE = ('PTS', 'REB', 'AST', 'FG3M')
Q1_STATS = tuple(f'{stat}_1Q' for stat in BASE)


def with_q1(history: pd.DataFrame, q1: pd.DataFrame) -> pd.DataFrame:
    """A copy of `history` with a _1Q column per stat.

    stats.nba.com lists only players who played in the quarter, so in a game the Q1 data covers, a missing
    player scored 0; a game it does not cover (an older season) stays empty rather than pretending 0.
    """
    out = history.copy()
    if q1.empty or not {'PLAYER_ID', 'GAME_ID'} <= set(q1):
        return out.assign(**{column: float('nan') for column in Q1_STATS})
    lines = q1.assign(_game=pd.to_numeric(q1['GAME_ID'], errors='coerce'), _player=pd.to_numeric(q1['PLAYER_ID'], errors='coerce'))
    lines = lines.rename(columns={stat: f'{stat}_1Q' for stat in BASE})[['_game', '_player', *Q1_STATS]]
    keys = out.assign(_game=pd.to_numeric(out['Game_ID'], errors='coerce'), _player=pd.to_numeric(out['Player_ID'], errors='coerce'))
    joined = keys[['_game', '_player']].merge(lines.drop_duplicates(['_game', '_player']), how='left', on=['_game', '_player'])
    covered = joined['_game'].isin(set(lines['_game'])).to_numpy()
    for column in Q1_STATS:
        values = pd.to_numeric(joined[column], errors='coerce').to_numpy()
        out[column] = pd.Series(values, index=out.index).where(~(covered & pd.isna(values)), 0)
    return out
