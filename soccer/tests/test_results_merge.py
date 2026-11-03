"""Merging football-data.org rows into the results lake.

football-data.co.uk stays canonical: it carries shots, cards, xG and odds, and
every existing team name in the lake is spelled its way. football-data.org is a
*gap filler* — it supplies the two or three days between a match finishing and
football-data.co.uk regenerating its season CSV, plus the Champions League
season openfootball never published.

So the rules under test are:
  1. A match present in both sources keeps the football-data.co.uk row.
  2. A football-data.org name that cannot be reconciled is dropped, not guessed.
  3. The Champions League is the exception — there is no other source, so its
     names are accepted as they come.
"""
import unittest

import polars as pl

from futball.teams import TeamResolver
from futball.transform import results


CANON_E0 = ["Arsenal", "Chelsea", "Nott'm Forest", "Tottenham", "Man United", "Man City"]


class NameReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.resolver = TeamResolver(CANON_E0)

    def test_football_data_org_names_resolve_onto_canonical_spellings(self):
        pairs = [
            ("Arsenal FC", "Arsenal"),
            ("Chelsea FC", "Chelsea"),
            ("Nottingham Forest FC", "Nott'm Forest"),
            ("Tottenham Hotspur FC", "Tottenham"),
            ("Manchester United FC", "Man United"),
            ("Manchester City FC", "Man City"),
        ]
        for foreign, canonical in pairs:
            with self.subTest(foreign=foreign):
                self.assertEqual(self.resolver.resolve(foreign), canonical)

    def test_an_unknown_club_does_not_resolve(self):
        self.assertIsNone(self.resolver.resolve("Real Madrid CF"))

    def test_hull_city_never_becomes_manchester_city(self):
        # The exact failure teams.py was written to prevent.
        self.assertNotEqual(self.resolver.resolve("Hull City AFC"), "Man City")


class StrictMergeTests(unittest.TestCase):
    def test_unresolvable_rows_are_dropped_in_strict_mode(self):
        rows = [
            {"date": "2026-09-12", "time": "14:00", "home_team": "Arsenal FC",
             "away_team": "Chelsea FC", "home_goals": 2, "away_goals": 1,
             "ht_home_goals": 1, "ht_away_goals": 0, "season": "2026/27"},
            {"date": "2026-09-12", "time": "16:30", "home_team": "Some New Club FC",
             "away_team": "Chelsea FC", "home_goals": 0, "away_goals": 0,
             "ht_home_goals": 0, "ht_away_goals": 0, "season": "2026/27"},
        ]
        out, dropped = results.reconcile_rows(
            rows, TeamResolver(CANON_E0), strict=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["home_team"], "Arsenal")
        self.assertEqual(dropped, ["Some New Club FC"])

    def test_dropped_names_are_reported_once_and_sorted(self):
        rows = [
            {"date": "2026-09-12", "time": None, "home_team": "Zeta Club FC",
             "away_team": "Alpha Club FC", "home_goals": 1, "away_goals": 0,
             "ht_home_goals": None, "ht_away_goals": None, "season": "2026/27"},
            {"date": "2026-09-13", "time": None, "home_team": "Zeta Club FC",
             "away_team": "Arsenal FC", "home_goals": 2, "away_goals": 2,
             "ht_home_goals": None, "ht_away_goals": None, "season": "2026/27"},
        ]
        out, dropped = results.reconcile_rows(rows, TeamResolver(CANON_E0), strict=True)
        self.assertEqual(out, [])
        self.assertEqual(dropped, ["Alpha Club FC", "Zeta Club FC"])

    def test_lenient_mode_keeps_unresolved_names(self):
        # Champions League: no canonical source exists, so a club appearing for
        # the first time must still show up.
        rows = [
            {"date": "2026-09-16", "time": "20:00", "home_team": "Qarabag FK",
             "away_team": "Arsenal FC", "home_goals": 0, "away_goals": 2,
             "ht_home_goals": 0, "ht_away_goals": 1, "season": "2026-27"},
        ]
        out, dropped = results.reconcile_rows(
            rows, TeamResolver(CANON_E0), strict=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["home_team"], "Qarabag FK")
        self.assertEqual(out[0]["away_team"], "Arsenal")  # still canonicalised
        self.assertEqual(dropped, [])


class DedupePrecedenceTests(unittest.TestCase):
    """The same match from both sources must collapse to the richer row."""

    def _frame(self, rows):
        return pl.DataFrame(rows, schema=results.ROW_SCHEMA)

    def test_football_data_couk_row_wins_over_football_data_org(self):
        key = "E0|2026/27|2026-09-12|Arsenal|Chelsea"
        rich = results._finish(results._row(
            match_key=key, competition_code="E0", competition="Premier League",
            country="England", season="2026/27", date="2026-09-12",
            home_team="Arsenal", away_team="Chelsea",
            home_goals=2, away_goals=1, home_shots=14, away_shots=9,
            odds_home=1.85))
        thin = results._finish(results._row(
            match_key=key, competition_code="E0", competition="Premier League",
            country="England", season="2026/27", date="2026-09-12",
            home_team="Arsenal", away_team="Chelsea",
            home_goals=2, away_goals=1))

        merged = results.dedupe(self._frame([rich, thin]))
        self.assertEqual(merged.height, 1)
        self.assertEqual(merged["home_shots"][0], 14, "richer row must survive")

    def test_order_does_not_matter_thin_row_first(self):
        key = "E0|2026/27|2026-09-12|Arsenal|Chelsea"
        thin = results._finish(results._row(
            match_key=key, competition_code="E0", season="2026/27",
            date="2026-09-12", home_team="Arsenal", away_team="Chelsea",
            home_goals=2, away_goals=1))
        rich = results._finish(results._row(
            match_key=key, competition_code="E0", season="2026/27",
            date="2026-09-12", home_team="Arsenal", away_team="Chelsea",
            home_goals=2, away_goals=1, home_shots=14))

        merged = results.dedupe(self._frame([thin, rich]))
        self.assertEqual(merged.height, 1)
        self.assertEqual(merged["home_shots"][0], 14)

    def test_equal_richness_is_broken_by_source_not_row_order(self):
        """Two rows can tie on richness. Precedence must still be deterministic.

        Before an explicit source rank this fell through to list order, so
        reordering the assembly in build() would silently have flipped which
        source won.
        """
        key = "E0|2026/27|2026-09-12|Arsenal|Chelsea"
        common = dict(
            match_key=key, competition_code="E0", season="2026/27",
            date="2026-09-12", home_team="Arsenal", away_team="Chelsea",
            home_goals=2, away_goals=1,
            time="14:00", ht_home_goals=1, ht_away_goals=0)
        canonical = results._finish(results._row(source="football-data.co.uk", **common))
        gap_filler = results._finish(results._row(source="football-data.org", **common))

        for label, ordered in (("canonical first", [canonical, gap_filler]),
                               ("gap-filler first", [gap_filler, canonical])):
            with self.subTest(order=label):
                merged = results.dedupe(self._frame(ordered))
                self.assertEqual(merged.height, 1)
                self.assertEqual(merged["source"][0], "football-data.co.uk")

    def test_openfootball_outranks_football_data_org_for_the_champions_league(self):
        key = "UCL|2026-27|2026-09-16|Arsenal|Chelsea"
        common = dict(
            match_key=key, competition_code="UCL", season="2026-27",
            date="2026-09-16", home_team="Arsenal", away_team="Chelsea",
            home_goals=1, away_goals=1, time="20:00",
            ht_home_goals=0, ht_away_goals=1)
        of_row = results._finish(results._row(source="openfootball", **common))
        fdo_row = results._finish(results._row(source="football-data.org", **common))
        merged = results.dedupe(self._frame([fdo_row, of_row]))
        self.assertEqual(merged.height, 1)
        self.assertEqual(merged["source"][0], "openfootball")

    def test_distinct_matches_are_both_kept(self):
        a = results._finish(results._row(
            match_key="E0|2026/27|2026-09-12|Arsenal|Chelsea",
            competition_code="E0", season="2026/27", date="2026-09-12",
            home_team="Arsenal", away_team="Chelsea", home_goals=2, away_goals=1))
        b = results._finish(results._row(
            match_key="E0|2026/27|2026-09-13|Man United|Man City",
            competition_code="E0", season="2026/27", date="2026-09-13",
            home_team="Man United", away_team="Man City", home_goals=1, away_goals=1))
        self.assertEqual(results.dedupe(self._frame([a, b])).height, 2)


if __name__ == "__main__":
    unittest.main()


class PromotedClubAliasTests(unittest.TestCase):
    """Clubs promoted into 2026/27, spelled in full by football-data.org.

    Both are present in the lake under football-data.co.uk's short form, so
    without an alias the resolver drops nine real La Liga matches — which is
    exactly what the refresh log reported before these were added.
    """

    CANON_SP1 = ["La Coruna", "Santander", "Barcelona", "Real Madrid", "Alaves"]

    def test_full_spanish_names_resolve_to_the_lake_spelling(self):
        resolver = TeamResolver(self.CANON_SP1)
        for foreign, canonical in [
            ("RC Deportivo La Coruña", "La Coruna"),
            ("Real Racing Club de Santander", "Santander"),
            ("Deportivo Alavés", "Alaves"),
        ]:
            with self.subTest(foreign=foreign):
                self.assertEqual(resolver.resolve(foreign), canonical)

    def test_santander_does_not_swallow_other_racing_clubs(self):
        # "racing" is a generic token shared by Lens, Santander, Strasbourg...
        resolver = TeamResolver(self.CANON_SP1 + ["Lens"])
        self.assertEqual(resolver.resolve("Racing Club de Lens"), "Lens")
