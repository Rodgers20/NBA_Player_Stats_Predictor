import pandas as pd

from utils.pregame_features import next_game_features


def games(rows):
    return pd.DataFrame([dict(PLAYER_NAME='A', GAME_DATE=day, SEASON=season, SEASON_ID=sid, PTS=pts, AST=3, REB=4, MIN=minutes,
                              FGA=10, FG_PCT=.5, FG3A=3, FG3_PCT=.3)
                         for day, season, sid, pts, minutes in rows])


REGULAR = [(f'2026-04-{day:02d}', '2025-26', 22025, 20, 32) for day in range(1, 11)]
PRESEASON = [('2026-10-05', '2026-27', 12026, 8, 18), ('2026-10-09', '2026-27', 12026, 6, 16)]


def test_preseason_games_do_not_feed_season_start_features():
    row = next_game_features(games(REGULAR + PRESEASON), '2026-10-21', True)
    assert row['rolling_avg_pts_5'] == 20
    assert row['rolling_avg_pts_10'] == 20
    assert row['season_avg_pts'] == 20  # the 2025-26 season, not two exhibition games


def test_offseason_gap_counts_as_normal_rest_not_a_long_layoff():
    row = next_game_features(games(REGULAR), '2026-10-21', True)
    assert row['days_rest'] == 2 and row['is_back_to_back'] == 0


def test_a_real_midseason_layoff_still_shows_as_rest():
    row = next_game_features(games(REGULAR), '2026-04-25', True)
    assert row['days_rest'] == 14


def test_preseason_only_history_is_still_usable():
    row = next_game_features(games(PRESEASON), '2026-10-21', True)
    assert row['rolling_avg_pts_5'] == 7
