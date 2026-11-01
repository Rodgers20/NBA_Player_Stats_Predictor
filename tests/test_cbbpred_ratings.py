import polars as pl

from cbbpred import ratings
from cbbpred.ratings import Params, Team
from tests.cbb_helpers import season_games

P = Params()


def test_winner_gains_and_loser_drops_on_both_ends():
    home, away = Team(0, 0, 2027), Team(0, 0, 2027)
    ph, pa = ratings.project(home, away, False, P)
    ratings.update(home, away, ph, pa, 90, 60, P)
    assert home.off > 0 and away.deff > 0      # home scored more than projected
    assert away.off < 0 and home.deff < 0      # away scored less than projected (home allowed less)
    assert home.games == away.games == 1


def test_error_is_capped():
    a, b = Team(0, 0, 2027, games=50), Team(0, 0, 2027, games=50)
    ratings.update(a, b, 70, 70, 200, 70, P)
    assert abs(a.off - P.k * P.cap) < 1e-9


def test_early_season_step_is_larger():
    early, late = Team(0, 0, 2027, games=0), Team(0, 0, 2027, games=30)
    other = Team(0, 0, 2027, games=30)
    ratings.update(early, other, 70, 70, 80, 70, P)
    other2 = Team(0, 0, 2027, games=30)
    ratings.update(late, other2, 70, 70, 80, 70, P)
    assert early.off > late.off


def test_neutral_site_removes_home_court():
    a, b = Team(0, 0, 1), Team(0, 0, 1)
    assert ratings.project(a, b, True, P) == (P.base, P.base)
    ph, pa = ratings.project(a, b, False, P)
    assert ph - pa == P.hfa


def test_non_d1_team_starts_below_average_and_season_carry_shrinks_ratings():
    games = season_games()
    pre, teams = ratings.run(games, {"2"}, P)          # conference 7 (team C) is "non-D1"
    first = pre.sort("game_id").row(0, named=True)
    assert first["home_off"] == 0.0                     # D1 team A starts at the mean
    c = next(r for r in pre.to_dicts() if r["home_off"] == P.unseen_off or r["away_off"] == P.unseen_off)
    assert c is not None
    teams["1"].off = 10.0
    ratings._roll_season(teams, {"1": True, "2": True, "3": False}, 2028, P)
    assert teams["1"].off == 10.0 * P.carry and teams["1"].games == 0


def test_projection_is_made_before_the_game_updates_anything():
    games = season_games()
    pre, _ = ratings.run(games, {"2", "7"}, P)
    first = pre.sort("game_id").row(0, named=True)
    assert first["rating_margin"] == P.hfa and first["home_gp"] == 0
    assert pre.filter(pl.col("game_id") == "1000")["home_gp"][0] == 0


def test_per100_scales_with_pace():
    assert abs(ratings.per100(0.0, 68.0) - 100.0) < 1e-9
