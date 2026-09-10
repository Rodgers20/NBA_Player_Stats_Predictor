"""Player features for an upcoming game, using completed observations only."""
from datetime import date
import pandas as pd


def prefer_identified_games(history):
    """Prefer a full game-ID observation over a partial duplicate live-feed row."""
    if 'Game_ID' not in history or 'PLAYER_NAME' not in history:
        return history
    history = history.copy()
    source = 'GAME_DATE' if 'GAME_DATE' in history else '_date'
    day = pd.to_datetime(history[source], format='mixed', errors='raise').dt.normalize()
    identified = pd.to_numeric(history['Game_ID'], errors='coerce').gt(0)
    any_identified = identified.groupby([history['PLAYER_NAME'], day]).transform('any')
    return history[identified | ~any_identified].copy()


def next_game_features(history, game_date=None, is_home=None):
    if history.empty:
        raise ValueError('No player history')
    history = prefer_identified_games(history).copy()
    source = 'GAME_DATE' if 'GAME_DATE' in history else '_date'
    history['_pregame_date'] = pd.to_datetime(history[source], format='mixed', errors='raise')
    when = pd.Timestamp(game_date or date.today()).normalize()
    history = history[history['_pregame_date'] < when].sort_values('_pregame_date', ascending=False)
    history = history.drop_duplicates('_pregame_date')
    if history.empty:
        raise ValueError('No completed games before the prediction date')
    row = history.iloc[0].to_dict()
    if is_home is not None:
        row['is_home'] = int(is_home)
    row['days_rest'] = min(14, max(0, (when - history.iloc[0]['_pregame_date']).days))
    row['is_back_to_back'] = int(row['days_rest'] <= 1)
    for stat in ('PTS', 'AST', 'REB', 'MIN', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT'):
        if stat not in history:
            continue
        values = pd.to_numeric(history[stat], errors='coerce')
        for window in (5, 10, 20):
            row[f'rolling_avg_{stat.lower()}_{window}'] = float(values.head(window).mean())
        if stat in ('PTS', 'AST', 'REB', 'MIN'):
            season = history[history['SEASON'].astype(str) == str(row['SEASON'])] if 'SEASON' in history else history
            row[f'season_avg_{stat.lower()}'] = float(pd.to_numeric(season[stat], errors='coerce').mean())
    if 'MIN' in history:
        minutes = pd.to_numeric(history['MIN'], errors='coerce')
        prior = minutes.iloc[1:6].mean()
        row['minutes_trend'] = float(minutes.head(5).mean() - prior) if pd.notna(prior) else 0.0
    return row
