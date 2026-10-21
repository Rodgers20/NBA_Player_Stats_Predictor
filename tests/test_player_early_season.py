import pandas as pd
from utils.early_season import player_season_games, shrink_projection

TARGET = pd.Timestamp('2026-10-21')


def rows(*items):
    return pd.DataFrame([dict(_date=pd.Timestamp(day), SEASON_ID=season) for day, season in items])


def test_player_season_games_ignores_preseason_and_last_season():
    history = rows(('2026-10-14', '12026'), ('2026-10-16', '12026'),  # preseason, reduced minutes
                   ('2026-06-10', '42025'), ('2026-03-06', '22025'),  # last season
                   ('2026-10-20', '22026'))                            # opening night
    assert player_season_games(history, TARGET) == 1


def test_player_season_games_excludes_the_target_day_and_later():
    assert player_season_games(rows(('2026-10-21', '22026'), ('2026-10-25', '22026')), TARGET) == 0


def test_projection_starts_at_the_market_line_and_trusts_the_model_as_games_arrive():
    assert shrink_projection(20.5, 24.5, 0) == 24.5
    assert shrink_projection(20.5, 24.5, 12) == 22.5   # half-trust at 12 games
    assert 20.5 < shrink_projection(20.5, 24.5, 60) < 21.5


def test_player_season_games_accepts_an_iso_date_string():
    assert player_season_games(rows(('2026-10-20', '22026')), '2026-10-21') == 1


def test_a_pick_needs_the_projection_to_favor_its_side():
    from api.routes.props import _serialise_prop
    base = dict(player='P', stat='PTS', line=24.5, ev=0.1, has_live_odds=True)
    assert _serialise_prop(dict(base, direction='Under', model_projection=20.5))['pick_type'] == 'pick'
    assert _serialise_prop(dict(base, direction='Under', model_projection=24.5))['pick_type'] == 'lean'
    assert _serialise_prop(dict(base, direction='Over', model_projection=20.5))['pick_type'] == 'lean'
