"""Qualification bands and the table's forward-looking Next column."""
import unittest

from futball import config, league


class BandTests(unittest.TestCase):
    def test_champions_league_league_phase_splits_at_8_and_24(self):
        """36 clubs: 1-8 straight to the last 16, 9-24 play off, 25-36 out."""
        self.assertEqual(config.qualification_band("UCL", 1)[0], "Qualification to 1/8 finals")
        self.assertEqual(config.qualification_band("UCL", 8)[0], "Qualification to 1/8 finals")
        self.assertEqual(config.qualification_band("UCL", 9)[0], "Knockout play-off round")
        self.assertEqual(config.qualification_band("UCL", 24)[0], "Knockout play-off round")
        self.assertEqual(config.qualification_band("UCL", 25)[0], "Eliminated")
        self.assertEqual(config.qualification_band("UCL", 36)[0], "Eliminated")

    def test_domestic_leagues_mark_europe_and_relegation(self):
        for code in ("E0", "SP1"):
            with self.subTest(code=code):
                self.assertIn("Champions League", config.qualification_band(code, 4)[0])
                self.assertIn("Europa", config.qualification_band(code, 5)[0])
                self.assertIn("Relegation", config.qualification_band(code, 20)[0])

    def test_mid_table_has_no_band(self):
        self.assertIsNone(config.qualification_band("E0", 10))
        self.assertIsNone(config.qualification_band("SP1", 12))

    def test_a_position_past_the_last_band_has_none(self):
        self.assertIsNone(config.qualification_band("E0", 99))

    def test_an_unmapped_competition_has_no_bands(self):
        self.assertIsNone(config.qualification_band("NOPE", 1))

    def test_every_band_that_carries_a_tone_names_itself(self):
        """A coloured bar with no legend entry is an unexplained decoration."""
        for code, bands in config.QUALIFICATION_BANDS.items():
            for last, label, tone in bands:
                with self.subTest(code=code, position=last):
                    if tone:
                        self.assertTrue(label, f"{code} position {last} has a tone but no label")

    def test_bands_are_ordered_and_cover_without_gaps(self):
        for code, bands in config.QUALIFICATION_BANDS.items():
            with self.subTest(code=code):
                cutoffs = [last for last, _l, _t in bands]
                self.assertEqual(cutoffs, sorted(cutoffs), f"{code} bands out of order")
                self.assertEqual(len(cutoffs), len(set(cutoffs)), f"{code} has a duplicate cutoff")


class StandingsIntegrationTests(unittest.TestCase):
    def test_every_row_carries_a_position_and_a_next_slot(self):
        table = league.standings("E0")
        if not table["rows"]:
            self.skipTest("no results cached")
        for row in table["rows"]:
            with self.subTest(team=row["team"]):
                self.assertIn("next", row)
                self.assertIn("qualification", row)

    def test_the_top_and_bottom_of_a_domestic_table_are_banded(self):
        table = league.standings("E0")
        if len(table["rows"]) < 20:
            self.skipTest("table not full")
        self.assertIsNotNone(table["rows"][0]["qualification"])
        self.assertIsNotNone(table["rows"][-1]["qualification"])

    def test_a_next_fixture_names_an_opponent_and_a_side(self):
        table = league.standings("E0")
        nexts = [r["next"] for r in table["rows"] if r.get("next")]
        if not nexts:
            self.skipTest("no fixtures cached")
        for n in nexts[:5]:
            self.assertTrue(n["opponent"])
            self.assertIsInstance(n["at_home"], bool)

    def test_a_club_is_never_its_own_next_opponent(self):
        for code in ("E0", "SP1", "UCL"):
            table = league.standings(code)
            for row in table["rows"]:
                if row.get("next"):
                    with self.subTest(code=code, team=row["team"]):
                        self.assertNotEqual(row["next"]["opponent"], row["team"])


if __name__ == "__main__":
    unittest.main()
