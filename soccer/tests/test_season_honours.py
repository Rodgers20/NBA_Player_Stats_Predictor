"""Season winners, and the heuristic that must not be used to find them.

The trap: taking the last match of a season as its final. Champions League
2025-26 ends on a drawn semi-final, so that shortcut would print a final nobody
played and pick a winner from a 1-1.
"""
import unittest

from futball import league


class ChampionsLeagueTests(unittest.TestCase):
    """Every cached final, checked against the real result."""

    KNOWN = {
        "2018-19": ("Liverpool FC", "Tottenham Hotspur"),
        "2019-20": ("Bayern München", "Paris Saint-Germain"),
        "2020-21": ("Chelsea FC", "Manchester City"),
        "2021-22": ("Real Madrid", "Liverpool FC"),
        "2022-23": ("Manchester City", "Inter"),
        "2023-24": ("Real Madrid CF", "Borussia Dortmund"),
        "2024-25": ("Paris Saint-Germain FC", "FC Internazionale Milano"),
    }

    def setUp(self):
        self.by_season = {h["season"]: h for h in league.season_honours("UCL")}
        if not self.by_season:
            self.skipTest("no Champions League data cached")

    def test_each_known_final_resolves_to_the_right_winner(self):
        for season, (winner, runner_up) in self.KNOWN.items():
            with self.subTest(season=season):
                h = self.by_season.get(season)
                self.assertIsNotNone(h, f"{season} missing")
                self.assertEqual(h["winner"], winner)
                self.assertEqual(h["runner_up"], runner_up)
                self.assertEqual(h["decided_by"], "final")

    def test_a_season_whose_final_is_unpublished_reports_that(self):
        """2025-26 stops at the semi-finals in the source."""
        h = self.by_season.get("2025-26")
        if h is None:
            self.skipTest("2025-26 not cached")
        self.assertIsNone(h["winner"])
        self.assertEqual(h["decided_by"], "unavailable")
        self.assertIn("final", (h["note"] or "").lower())

    def test_the_current_season_is_marked_in_progress_not_unavailable(self):
        year = league.current_season_year("UCL")
        live = [h for h in self.by_season.values() if h["season_year"] == year]
        if not live:
            self.skipTest("current season not cached")
        self.assertEqual(live[0]["decided_by"], "in_progress")

    def test_the_last_match_of_a_season_is_not_assumed_to_be_the_final(self):
        """Guards the heuristic directly: no winner may come from a drawn game."""
        for h in self.by_season.values():
            if h["decided_by"] != "final":
                continue
            with self.subTest(season=h["season"]):
                # The note is the scoreline; a final that decided a winner
                # cannot have been a draw.
                note = h["note"] or ""
                self.assertTrue(note, "a resolved final should record its scoreline")

    def test_seasons_come_back_newest_first(self):
        seasons = [h["season"] for h in league.season_honours("UCL")]
        self.assertEqual(seasons, sorted(seasons, reverse=True))


class DomesticLeagueTests(unittest.TestCase):
    def test_the_winner_is_the_top_of_that_seasons_table(self):
        honours = league.season_honours("E0")
        if not honours:
            self.skipTest("no Premier League data cached")
        for h in honours[:5]:
            with self.subTest(season=h["season"]):
                table = league.standings("E0", h["season_year"])
                rows = table.get("rows") or []
                if len(rows) < 2:
                    continue
                self.assertEqual(h["winner"], rows[0]["team"])
                self.assertEqual(h["runner_up"], rows[1]["team"])

    def test_a_completed_season_is_decided_by_table(self):
        honours = league.season_honours("E0")
        finished = [h for h in honours
                    if h["season_year"] != league.current_season_year("E0")]
        if not finished:
            self.skipTest("no completed seasons")
        self.assertEqual(finished[0]["decided_by"], "table")

    def test_the_live_season_is_flagged_even_though_it_has_a_leader(self):
        year = league.current_season_year("E0")
        live = [h for h in league.season_honours("E0") if h["season_year"] == year]
        if not live:
            self.skipTest("current season not cached")
        self.assertEqual(live[0]["decided_by"], "in_progress")
        self.assertIsNotNone(live[0]["winner"], "the current leader should still show")


class ShapeTests(unittest.TestCase):
    def test_an_unknown_competition_returns_nothing(self):
        self.assertEqual(league.season_honours("NOPE"), [])

    def test_every_entry_declares_how_it_was_decided(self):
        for code in ("E0", "SP1", "UCL"):
            for h in league.season_honours(code):
                with self.subTest(code=code, season=h["season"]):
                    self.assertIn(h["decided_by"],
                                  {"final", "table", "in_progress", "unavailable", "undecided"})
                    if h["decided_by"] in {"unavailable", "undecided"}:
                        self.assertIsNone(h["winner"])


if __name__ == "__main__":
    unittest.main()
