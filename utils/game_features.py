"""Pregame team features reconstructed from complete reciprocal player box scores.

Every feature uses games strictly before the target calendar date. Same-day games
are intentionally excluded; injuries and other unavailable historical news are
never reconstructed from present-day feeds.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
import numpy as np
import pandas as pd

_STATS = ('MIN', 'PTS', 'FGA', 'FGM', 'FG3M', 'FTA', 'FTM', 'OREB', 'DREB', 'TOV')
# Arena-city approximations (latitude, longitude, UTC standard offset, elevation m).
# Deliberately sparse: unknown venues yield explicit missing indicators.
_ARENAS = {
    'DEN': (39.749, -105.008, -7, 1609), 'UTA': (40.768, -111.901, -7, 1288),
    'PHX': (33.446, -112.071, -7, 331), 'LAL': (34.043, -118.267, -8, 89),
    'BOS': (42.366, -71.062, -5, 6), 'NYK': (40.751, -73.994, -5, 10),
    'CHI': (41.880, -87.674, -6, 181), 'DAL': (32.790, -96.810, -6, 131),
    'MIA': (25.781, -80.188, -5, 2), 'GSW': (37.768, -122.388, -8, 4),
}


def _abbr(value, league):
    value = str(value).upper().strip()
    aliases = {'NY': 'NYK', 'GS': 'GSW', 'SA': 'SAS', 'NO': 'NOP', 'UTAH': 'UTA',
               'WSH': 'WAS', 'NJ': 'BKN', 'PHO': 'PHX'}
    if league == 'wnba':
        aliases.update({'NY': 'NYL', 'GS': 'GSV', 'GSW': 'GSV', 'LA': 'LAS', 'LV': 'LVA', 'POR': 'PDX'})
    return aliases.get(value, value)


def _regulation(league):
    if league not in ('nba', 'wnba'):
        raise ValueError('league must be nba or wnba')
    return 48 if league == 'nba' else 40


def aggregate_team_games(player_logs: pd.DataFrame, league='nba') -> pd.DataFrame:
    """Return two rows per complete game, deduplicating player/game/team records.

    Input uses nba_api box-score names (case-insensitive). NBA preseason and
    exhibition season IDs are excluded when SEASON_ID exists. Playoffs remain
    eligible and are flagged separately.
    """
    regulation = _regulation(league)
    frame = player_logs.copy()
    frame.columns = frame.columns.str.upper()
    required = {'GAME_ID', 'GAME_DATE', 'PLAYER_ID', 'TEAM_ABBREVIATION', 'MATCHUP'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing player log columns: {sorted(required - set(frame.columns))}')
    frame['date'] = pd.to_datetime(frame['GAME_DATE'], errors='coerce', format='mixed').dt.normalize()
    frame['game_id'] = frame['GAME_ID'].astype(str)
    team_codes = frame['TEAM_ABBREVIATION'].replace('', np.nan).fillna(frame['MATCHUP'].str.split().str[0])
    frame['team'] = team_codes.map(lambda x: _abbr(x, league))
    frame['opponent'] = frame['MATCHUP'].str.split().str[-1].map(lambda x: _abbr(x, league))
    frame['home'] = frame['MATCHUP'].str.contains('vs', case=False, na=False)
    if 'SEASON_ID' in frame:
        sid = frame['SEASON_ID'].astype(str)
        frame = frame[sid.str.startswith(('2', '4'))].copy()
        frame['playoffs'] = frame['SEASON_ID'].astype(str).str.startswith('4').astype(int)
    else:
        frame['playoffs'] = 0
    frame = frame.dropna(subset=['date']).drop_duplicates(['game_id', 'team', 'PLAYER_ID'])
    for stat in _STATS:
        if stat not in frame:
            frame[stat] = np.nan
        if stat == 'MIN':
            def minutes(value):
                if isinstance(value, str) and ':' in value:
                    try:
                        parts = value.split(':')
                        return float(parts[0]) + float(parts[1]) / 60
                    except ValueError:
                        return np.nan
                return pd.to_numeric(value, errors='coerce')
            frame[stat] = frame[stat].map(minutes)
        else:
            frame[stat] = pd.to_numeric(frame[stat], errors='coerce')
    # DNP rows have no minutes and no scoring contribution. A scoring row with
    # missing minutes stays present so the completeness check rejects it.
    frame = frame[~((frame['MIN'].isna() | (frame['MIN'] <= 0)) &
                    frame['PTS'].fillna(0).eq(0))].copy()
    grouped = frame.groupby(['game_id', 'team'], sort=False)
    totals = grouped[list(_STATS)].sum(min_count=1)
    # A partially missing stat is unknown, rather than a silently understated sum.
    for stat in _STATS:
        totals.loc[grouped[stat].count() != grouped.size(), stat] = np.nan
    metadata = grouped[['date', 'opponent', 'home', 'playoffs']].first()
    result = metadata.join(totals).reset_index()
    result = result[result['MIN'] >= regulation * 5 * .90].copy()
    valid = []
    for _, pair in result.groupby('game_id', sort=False):
        if len(pair) != 2:
            continue
        a, b = pair.iloc[0], pair.iloc[1]
        if (a['opponent'] == b['team'] and b['opponent'] == a['team']
                and a['date'] == b['date'] and bool(a['home']) != bool(b['home'])
                and pd.notna(a['PTS']) and pd.notna(b['PTS'])):
            valid.extend(pair.index)
    result = result.loc[valid].copy()
    if result.empty:
        return result
    opponent = result.set_index(['game_id', 'team'])
    for stat in _STATS:
        result[f'opp_{stat}'] = [opponent.loc[(g, o), stat] for g, o in zip(result.game_id, result.opponent)]
    result['possessions'] = result.FGA + .44 * result.FTA - result.OREB + result.TOV
    result['opp_possessions'] = result.opp_FGA + .44 * result.opp_FTA - result.opp_OREB + result.opp_TOV
    possessions = (result.possessions + result.opp_possessions) / 2
    safe = possessions.where(possessions > 0)
    result['offense'] = 100 * result.PTS / safe
    result['defense'] = 100 * result.opp_PTS / safe
    result['pace'] = safe * regulation * 5 / result.MIN
    result['efg'] = (result.FGM + .5 * result.FG3M) / result.FGA.where(result.FGA > 0)
    result['tov_rate'] = result.TOV / result.possessions.where(result.possessions > 0)
    result['oreb_share'] = result.OREB / (result.OREB + result.opp_DREB).where(result.OREB + result.opp_DREB > 0)
    result['ft_rate'] = result.FTA / result.FGA.where(result.FGA > 0)
    result['margin'] = result.PTS - result.opp_PTS
    result['venue_team'] = np.where(result.home, result.team, result.opponent)
    result['season'] = result.date.dt.year - ((result.date.dt.month < 7) & (league == 'nba')).astype(int)
    return result.sort_values(['date', 'game_id', 'team']).reset_index(drop=True)


def _elo(history):
    ratings = {}
    season = None
    for _, pair in history.groupby(['date', 'game_id'], sort=True):
        if len(pair) != 2:
            continue
        current_season = pair.iloc[0]['season']
        if current_season != season:
            ratings = {team: 1500 + .75 * (rating - 1500) for team, rating in ratings.items()}
            season = current_season
        h = pair[pair.home].iloc[0]
        a = pair[~pair.home].iloc[0]
        hr, ar = ratings.get(h.team, 1500), ratings.get(a.team, 1500)
        expected = 1 / (1 + 10 ** ((ar - hr - 65) / 400))
        outcome = float(h.margin > 0) if h.margin != 0 else .5
        ratings[h.team] = hr + 20 * (outcome - expected)
        ratings[a.team] = ar - 20 * (outcome - expected)
    return ratings


def _distance(a, b):
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    return 6371 * 2 * asin(sqrt(sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2))


def build_feature_vector(team_games, target_date, home_team, away_team, league='nba', neutral=False, venue=None, _ratings=None):
    """Build the identical flat numeric vector for historical or future games.

    ``venue`` is an optional canonical arena team code. Coordinates are only
    available for documented NBA arena-city approximations above. Standard UTC
    offsets are used as travel-zone proxies, not local tipoff timestamps.
    """
    _regulation(league)
    target = pd.Timestamp(target_date).normalize()
    history = team_games[team_games.date < target].copy()
    ratings = _elo(history) if _ratings is None else _ratings
    teams = [_abbr(home_team, league), _abbr(away_team, league)]
    features = {'neutral': int(neutral), 'home_advantage': int(not neutral)}
    destination = _ARENAS.get(_abbr(venue, league) if venue else teams[0]) if league == 'nba' and (venue or not neutral) else None
    metrics = ('offense', 'defense', 'pace', 'efg', 'tov_rate', 'oreb_share', 'ft_rate', 'margin')
    season = target.year - int(league == 'nba' and target.month < 7)
    for side, team in zip(('home', 'away'), teams):
        logs = history[history.team == team].sort_values(['date', 'game_id'])
        features[f'{side}_history_games'] = len(logs)
        features[f'{side}_history_missing'] = int(logs.empty)
        features[f'{side}_elo'] = ratings.get(team, 1500.)
        features[f'{side}_opponent_elo_last10'] = float(np.mean([ratings.get(x, 1500.) for x in logs.tail(10).opponent])) if len(logs) else 1500.
        for window in (5, 10, 20):
            for metric in metrics:
                value = logs.tail(window)[metric].mean() if len(logs) else np.nan
                features[f'{side}_{metric}_{window}'] = value
                features[f'{side}_{metric}_{window}_missing'] = int(pd.isna(value))
        for location, is_home in (('home', True), ('away', False)):
            local = logs[logs.home == is_home].tail(10)
            features[f'{side}_{location}_margin_10'] = local.margin.mean() if len(local) else np.nan
            features[f'{side}_{location}_form_missing'] = int(local.empty)
        seasonal = logs[logs.season == season]
        for metric in ('offense', 'defense', 'pace', 'margin'):
            prior = history[metric].mean() if len(history) else np.nan
            observed = seasonal[metric].dropna()
            features[f'{side}_{metric}_season_regressed'] = (observed.sum() + 10 * prior) / (len(observed) + 10) if pd.notna(prior) else np.nan
            features[f'{side}_{metric}_season_missing'] = int(pd.isna(features[f'{side}_{metric}_season_regressed']))
        rest = (target - logs.iloc[-1].date).days - 1 if len(logs) else np.nan
        features[f'{side}_rest_days'] = rest
        features[f'{side}_rest_missing'] = int(pd.isna(rest))
        features[f'{side}_b2b'] = int(rest == 0) if pd.notna(rest) else np.nan
        for days in (4, 7):
            features[f'{side}_games_last{days}'] = int((logs.date >= target - pd.Timedelta(days=days)).sum())
        origin = _ARENAS.get(logs.iloc[-1].venue_team) if len(logs) and league == 'nba' else None
        missing = origin is None or destination is None
        features[f'{side}_travel_missing'] = int(missing)
        features[f'{side}_travel_km'] = np.nan if missing else _distance(origin, destination)
        features[f'{side}_timezone_change'] = np.nan if missing else destination[2] - origin[2]
        features[f'{side}_altitude_change_m'] = np.nan if missing else destination[3] - origin[3]
    for side in ('home', 'away'):
        for metric in ('offense', 'defense'):
            features[f'{side}_{metric}'] = features[f'{side}_{metric}_10']
        features[f'{side}_recent_margin'] = features[f'{side}_margin_10']
        features[f'{side}_pace'] = features[f'{side}_pace_10']
    features['elo_diff'] = features['home_elo'] - features['away_elo']
    return features


BASELINE_FEATURE_COLUMNS = ('home_offense', 'away_offense', 'home_defense', 'away_defense',
                            'elo_diff', 'home_recent_margin', 'away_recent_margin')
# Metadata and targets are intentionally excluded.
FEATURE_COLUMNS = tuple(build_feature_vector(
    pd.DataFrame(columns=['date', 'team', 'season', 'opponent', 'home', 'game_id', 'offense', 'defense', 'pace', 'margin']),
    '2026-10-01', 'LAL', 'BOS'))


def normalize_team_games(team_logs, league='nba'):
    """Accept authoritative summed NBA-compatible team box scores."""
    frame = team_logs.copy()
    frame.columns = frame.columns.str.upper()
    if 'SEASON_TYPE' in frame:
        frame = frame[frame.SEASON_TYPE.astype(str).str.lower().isin(
            ['regular', 'regular season', 'playoffs', 'postseason', '2', '4'])].copy()
    frame['PLAYER_ID'] = frame['TEAM_ABBREVIATION']
    return aggregate_team_games(frame, league)


def build_game_features(player_logs, league='nba'):
    """Return one row per reciprocal game; targets are separate from features."""
    games = aggregate_team_games(player_logs, league)
    return build_matchup_features(games, league)


def build_matchup_features(games, league='nba'):
    """Build features from canonical aggregated team games, updating Elo by day."""
    rows, ratings, season = [], {}, None
    for day, daily in games.groupby('date', sort=True):
        for _, pair in daily.groupby('game_id', sort=True):
            home, away = pair[pair.home].iloc[0], pair[~pair.home].iloc[0]
            features = build_feature_vector(games, day, home.team, away.team, league, _ratings=ratings)
            rows.append(dict(game_id=home.game_id, date=day, home_team=home.team,
                             away_team=away.team, playoffs=int(home.playoffs),
                             margin=float(home.margin), total=float(home.PTS + away.PTS), **features))
        # Commit outcomes only after every feature vector for this day exists.
        for _, pair in daily.groupby('game_id', sort=True):
            h, a = pair[pair.home].iloc[0], pair[~pair.home].iloc[0]
            if h.season != season:
                ratings = {team: 1500 + .75 * (r - 1500) for team, r in ratings.items()}
                season = h.season
            hr, ar = ratings.get(h.team, 1500.), ratings.get(a.team, 1500.)
            expected = 1 / (1 + 10 ** ((ar - hr - 65) / 400))
            outcome = float(h.margin > 0) if h.margin != 0 else .5
            ratings[h.team] = hr + 20 * (outcome - expected)
            ratings[a.team] = ar - 20 * (outcome - expected)
    return pd.DataFrame(rows)
