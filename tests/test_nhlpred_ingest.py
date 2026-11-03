import pandas as pd

from nhlpred.ingest import _with_shots


def test_shots_survive_a_rescan_and_fall_back_to_skater_sums():
    rescanned = pd.DataFrame({"game_id": [1, 2, 3], "date": ["d1", "d2", "d3"], "start_utc": ["a", "b", "c"]})   # schedule rows carry no shots
    old = pd.DataFrame({"game_id": [1], "home_sog": [31.0], "away_sog": [22.0]})
    sk = pd.DataFrame({"game_id": [2, 2, 2], "home": [True, True, False], "sog": [3, 4, 5]})
    out = _with_shots(rescanned, old, {3: {"home_sog": 40, "away_sog": 20}}, sk).set_index("game_id")
    assert (out.loc[1, "home_sog"], out.loc[1, "away_sog"]) == (31, 22)      # kept from the stored table
    assert (out.loc[2, "home_sog"], out.loc[2, "away_sog"]) == (7, 5)        # rebuilt from skaters
    assert (out.loc[3, "home_sog"], out.loc[3, "away_sog"]) == (40, 20)      # fresh boxscore
