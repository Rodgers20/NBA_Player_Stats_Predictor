from datetime import date, timedelta
from unittest.mock import patch

import polars as pl

from futball import league
from futball.predict import matchday


def fixtures_frame():
    today = date.today()
    day = lambda n: (today + timedelta(days=n)).isoformat()  # noqa: E731
    return pl.DataFrame({
        "competition_code": ["MLS", "MLS", "MLS", "MLS", "E0"],
        "date": [day(-5), day(-1), day(2), day(30), day(2)],
        "time": ["20:00"] * 5, "kickoff_utc": ["k"] * 5,
        "home_team": ["Old", "Austin FC", "Inter Miami", "Far", "Arsenal"],
        "away_team": ["Gone", "FC Dallas", "Charlotte", "Away", "Chelsea"],
        "home_goals": [1, 2, None, None, None], "away_goals": [0, 2, None, None, None],
    })


def test_competitions_without_provider_rounds_get_a_rolling_window_of_fixtures():
    with patch.object(league, "_fixtures", return_value=fixtures_frame()):
        rows = matchday._window_fixtures("MLS")
    assert [(r["home_team"], r["status"]) for r in rows] == [("Austin FC", "FINISHED"), ("Inter Miami", "SCHEDULED")]
    assert rows[0]["home_goals"] == 2 and rows[1]["home_goals"] is None


def test_board_for_such_a_competition_has_no_matchday_number_but_lists_the_window():
    with patch.object(league, "_fixtures", return_value=fixtures_frame()), \
         patch.object(matchday.serve, "teams", return_value=[]):
        board = matchday.board("MLS")
    assert board["matchday"] is None and board["matchdays"] == []
    assert [f["home_team"] for f in board["fixtures"]] == ["Austin FC", "Inter Miami"]
    assert board["fixtures"][0]["unavailable"] == "played"
