"""The round-at-a-glance board on the predictions page."""
import unittest
from unittest.mock import patch

from futball import league
from futball.predict import matchday, serve


def _skip_without_models(test):
    try:
        serve._bundle()
    except serve.ModelsUnavailable:
        test.skipTest("models not trained")


MATCHES = {"matches": [
    {"id": 1, "matchday": 6, "status": "FINISHED", "utcDate": "2026-09-13T15:00:00Z",
     "homeTeam": {"name": "FC Barcelona"}, "awayTeam": {"name": "Getafe CF"},
     "score": {"fullTime": {"home": 3, "away": 0}}},
    {"id": 2, "matchday": 7, "status": "TIMED", "utcDate": "2026-09-19T19:00:00Z",
     "homeTeam": {"name": "Real Madrid CF"}, "awayTeam": {"name": "Sevilla FC"},
     "score": {"fullTime": {"home": None, "away": None}}},
    {"id": 3, "matchday": 7, "status": "SCHEDULED", "utcDate": "2026-09-20T17:00:00Z",
     "homeTeam": {"name": "Valencia CF"}, "awayTeam": {"name": "Villarreal CF"},
     "score": {"fullTime": {"home": None, "away": None}}},
    {"id": 4, "matchday": 8, "status": "SCHEDULED", "utcDate": "2026-09-27T17:00:00Z",
     "homeTeam": {"name": "FC Barcelona"}, "awayTeam": {"name": "Real Madrid CF"},
     "score": {"fullTime": {"home": None, "away": None}}},
]}


class MatchdaySelectionTests(unittest.TestCase):
    def test_the_current_round_is_the_first_with_anything_left(self):
        with patch.object(matchday, "_matches", return_value=MATCHES["matches"]):
            self.assertEqual(matchday.current_matchday("SP1"), 7)

    def test_a_fully_played_season_has_no_current_round(self):
        played = [{**m, "status": "FINISHED"} for m in MATCHES["matches"]]
        with patch.object(matchday, "_matches", return_value=played):
            self.assertIsNone(matchday.current_matchday("SP1"))

    def test_every_round_is_offered_for_paging(self):
        with patch.object(matchday, "_matches", return_value=MATCHES["matches"]):
            self.assertEqual(matchday.available_matchdays("SP1"), [6, 7, 8])


class BoardTests(unittest.TestCase):
    def setUp(self):
        _skip_without_models(self)
        from futball import clubs
        if not clubs.directory():
            self.skipTest("club directory not cached")

    def _board(self, md=None):
        with patch.object(matchday, "_matches", return_value=MATCHES["matches"]):
            return matchday.board("SP1", md)

    def test_it_defaults_to_the_round_in_progress(self):
        self.assertEqual(self._board()["matchday"], 7)

    def test_only_that_round_is_listed(self):
        board = self._board()
        self.assertEqual(len(board["fixtures"]), 2)

    def test_every_upcoming_fixture_carries_a_forecast(self):
        for f in self._board()["fixtures"]:
            with self.subTest(fixture=f"{f['home_team']} v {f['away_team']}"):
                self.assertIsNotNone(f["prediction"], f["unavailable"])

    def test_a_forecast_is_internally_consistent(self):
        for f in self._board()["fixtures"]:
            p = f["prediction"]
            with self.subTest(fixture=f["home_team"]):
                self.assertAlmostEqual(sum(p["win"].values()), 1.0, places=3)
                self.assertEqual(p["result_pick"], max(p["win"], key=p["win"].get))
                self.assertTrue(0.0 <= p["over_2_5"] <= 1.0)

    def test_a_played_fixture_shows_its_result_and_no_forecast(self):
        board = self._board(6)
        played = board["fixtures"][0]
        self.assertEqual(played["unavailable"], "played")
        self.assertIsNone(played["prediction"])
        self.assertEqual((played["home_goals"], played["away_goals"]), (3, 0))

    def test_fixtures_run_in_kick_off_order(self):
        kickoffs = [f["kickoff_utc"] for f in self._board()["fixtures"]]
        self.assertEqual(kickoffs, sorted(kickoffs))

    def test_provider_names_are_resolved_to_the_lake(self):
        home = {f["home_team"] for f in self._board()["fixtures"]}
        self.assertIn("Real Madrid", home)

    def test_the_note_counts_what_was_forecast(self):
        self.assertIn("2 of 2 fixtures forecast", self._board()["note"])


class RealCompetitionTests(unittest.TestCase):
    """Against the live fixture cache, not a fixture."""

    def setUp(self):
        _skip_without_models(self)
        from futball import clubs
        if not clubs.directory():
            self.skipTest("club directory not cached")

    def test_each_competition_forecasts_its_whole_round(self):
        for code in ("E0", "SP1", "UCL"):
            board = matchday.board(code)
            if not board["fixtures"]:
                continue
            forecast = sum(1 for f in board["fixtures"] if f["prediction"])
            unplayed = sum(1 for f in board["fixtures"] if f["unavailable"] != "played")
            with self.subTest(code=code, matchday=board["matchday"]):
                self.assertEqual(forecast, unplayed,
                                 [f"{f['home_team']} v {f['away_team']}: {f['unavailable']}"
                                  for f in board["fixtures"] if not f["prediction"]])

    def test_the_board_is_the_same_on_every_call(self):
        """Team selection used to move with Python's hash seed between runs."""
        first = [f["unavailable"] for f in matchday.board("UCL")["fixtures"]]
        second = [f["unavailable"] for f in matchday.board("UCL")["fixtures"]]
        self.assertEqual(first, second)


class SpellingTests(unittest.TestCase):
    """One club, one lake spelling per competition — agreed by both layers."""

    def setUp(self):
        from futball import clubs
        if not clubs.directory():
            self.skipTest("club directory not cached")
        _skip_without_models(self)

    def test_resolution_matches_what_the_model_will_accept(self):
        for code in ("E0", "SP1", "UCL"):
            known = set(serve.teams(code))
            for name in (f["home_team"] for f in matchday.board(code)["fixtures"]):
                if name is None:
                    continue
                with self.subTest(code=code, club=name):
                    self.assertIn(name, known)

    def test_the_current_season_spelling_wins_over_the_historic_one(self):
        self.assertEqual(league._to_lake_name("Real Madrid CF", "UCL"), "Real Madrid CF")
        self.assertEqual(league._to_lake_name("Real Madrid CF", "SP1"), "Real Madrid")


if __name__ == "__main__":
    unittest.main()
