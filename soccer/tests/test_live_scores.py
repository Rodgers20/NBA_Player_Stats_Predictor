"""Running scores on the league table.

The live feed is ESPN (`futball/live/espn.py`): it publishes a real status per
match, so nothing here infers "in play" from a kick-off time plus a duration.
These tests cover the parse, the name reconciliation that lets a provider's
club reach the lake's spelling, and the score shape the table consumes.
"""
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from futball import league
from futball.live import espn

NOW = datetime(2026, 9, 15, 18, 30, tzinfo=timezone.utc)


def _event(eid, home, away, state, name, home_score, away_score, kickoff, completed=False):
    return {
        "id": eid,
        "date": kickoff,
        "competitions": [{
            "status": {"type": {"name": name, "state": state, "completed": completed,
                                "shortDetail": "70'"}},
            "competitors": [
                {"homeAway": "home", "team": {"displayName": home}, "score": home_score},
                {"homeAway": "away", "team": {"displayName": away}, "score": away_score},
            ],
        }],
    }


BOARD = {"events": [
    _event("1", "Rayo Vallecano", "Espanyol", "in", "STATUS_IN_PROGRESS",
           "2", "1", "2026-09-15T17:00Z"),
    _event("2", "Deportivo Alaves", "Valencia", "in", "STATUS_HALFTIME",
           "0", "0", "2026-09-15T18:00Z"),
    _event("3", "Barcelona", "Getafe", "post", "STATUS_FINAL",
           "3", "0", "2026-09-15T15:00Z", completed=True),
    _event("4", "Sevilla", "Villarreal", "pre", "STATUS_SCHEDULED",
           None, None, "2026-09-15T20:00Z"),
    _event("5", "Levante", "Osasuna", "in", "STATUS_POSTPONED",
           None, None, "2026-09-15T19:00Z"),
]}


class ParseTests(unittest.TestCase):
    def setUp(self):
        from futball import clubs
        if not clubs.directory():
            self.skipTest("club directory not cached; run `make refresh`")
        self.rows = {r["provider_id"]: r for r in espn.parse(BOARD, "SP1", NOW)}

    def test_an_in_progress_match_is_live(self):
        self.assertTrue(self.rows["1"]["is_live"])

    def test_half_time_is_still_live(self):
        self.assertTrue(self.rows["2"]["is_live"])

    def test_a_finished_match_is_not_live_but_keeps_its_score(self):
        row = self.rows["3"]
        self.assertFalse(row["is_live"])
        self.assertEqual((row["home_goals"], row["away_goals"]), (3, 0))

    def test_a_scheduled_match_has_no_score(self):
        row = self.rows["4"]
        self.assertFalse(row["is_live"])
        self.assertIsNone(row["home_goals"])

    def test_a_postponed_match_is_not_live_however_it_is_labelled(self):
        """ESPN reports state "in" on a postponement; it is not under way."""
        self.assertFalse(self.rows["5"]["is_live"])

    def test_scores_keep_home_and_away_the_right_way_round(self):
        row = self.rows["1"]
        self.assertEqual((row["home_goals"], row["away_goals"]), (2, 1))

    def test_provider_names_are_reconciled_to_the_lake(self):
        row = self.rows["1"]
        self.assertEqual(row["home_team"], "Vallecano")
        self.assertEqual(row["away_team"], "Espanol")

    def test_an_empty_board_is_not_an_error(self):
        self.assertEqual(espn.parse({}, "SP1", NOW), [])
        self.assertEqual(espn.parse({"events": []}, "SP1", NOW), [])

    def test_a_duplicate_event_appears_once(self):
        doubled = {"events": BOARD["events"] + [BOARD["events"][0]]}
        rows = espn.parse(doubled, "SP1", NOW)
        ids = [r["provider_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))


class NameResolutionTests(unittest.TestCase):
    """Provider long names must reach the lake's own spelling."""

    def setUp(self):
        from futball import clubs
        if not clubs.directory():
            self.skipTest("club directory not cached")

    def test_provider_names_reach_the_lake(self):
        for provider, expected in (("Rayo Vallecano de Madrid", "Vallecano"),
                                   ("RCD Espanyol de Barcelona", "Espanol"),
                                   ("Valencia CF", "Valencia")):
            with self.subTest(provider=provider):
                self.assertEqual(league._to_lake_name(provider, "SP1"), expected)

    def test_the_index_is_scoped_to_one_competition(self):
        """The lake writes "Valencia" in La Liga and "Valencia CF" in the UCL."""
        spanish = league._lake_name_by_club("SP1")
        european = league._lake_name_by_club("UCL")
        club = "Valencia CF"
        if club in spanish and club in european:
            self.assertNotEqual(spanish[club], european[club])

    def test_an_unknown_club_resolves_to_nothing(self):
        self.assertIsNone(league._to_lake_name("Not A Club FC", "SP1"))
        self.assertIsNone(league._to_lake_name(None, "SP1"))


class TableScoreTests(unittest.TestCase):
    """What the standings chips actually consume."""

    def setUp(self):
        from futball import clubs
        if not clubs.directory():
            self.skipTest("club directory not cached")
        rows = espn.parse(BOARD, "SP1", NOW)
        self.snapshot = {
            "total": sum(r["is_live"] for r in rows),
            "matches": [{**r, "stale": False} for r in rows if r["is_live"]],
            "by_competition": {"SP1": sum(r["is_live"] for r in rows)},
            "stale": False, "source": "ESPN",
        }

    def test_only_scored_live_matches_become_chips(self):
        with patch.object(espn, "counts", return_value=self.snapshot):
            scores = league.live_scores("SP1")
        self.assertIn("SP1|Vallecano|Espanol", scores)
        self.assertEqual(scores["SP1|Vallecano|Espanol"]["home_goals"], 2)

    def test_half_time_is_reported_as_paused(self):
        with patch.object(espn, "counts", return_value=self.snapshot):
            scores = league.live_scores("SP1")
        self.assertEqual(scores["SP1|Alaves|Valencia"]["status"], "PAUSED")

    def test_a_match_without_a_score_is_not_a_chip(self):
        """An in-play match whose score has not landed must not read 0-0."""
        blank = {**self.snapshot, "matches": [
            {**m, "home_goals": None, "away_goals": None} for m in self.snapshot["matches"]
        ]}
        with patch.object(espn, "counts", return_value=blank):
            self.assertEqual(league.live_scores("SP1"), {})

    def test_one_match_appears_once(self):
        with patch.object(espn, "counts", return_value=self.snapshot):
            rows = league.live_now("SP1")
        pairings = [(r["home_team"], r["away_team"]) for r in rows]
        self.assertEqual(len(pairings), len(set(pairings)), pairings)

    def test_another_competition_is_filtered_out(self):
        with patch.object(espn, "counts", return_value=self.snapshot):
            self.assertEqual(league.live_now("E0"), [])


if __name__ == "__main__":
    unittest.main()
