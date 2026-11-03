"""Team colours must resolve across every spelling the three sources use."""
import unittest

import polars as pl

from futball import config, teamcolors


class ResolutionTests(unittest.TestCase):
    def test_the_same_club_resolves_from_every_source_spelling(self):
        groups = [
            ("Man City", "Manchester City FC", "Manchester City"),
            ("Barcelona", "FC Barcelona", "Barça"),
            ("Real Madrid", "Real Madrid CF"),
            ("Ath Madrid", "Club Atlético de Madrid", "Atletico Madrid"),
            ("Paris SG", "Paris Saint-Germain FC"),
            ("Nott'm Forest", "Nottingham Forest FC"),
        ]
        for spellings in groups:
            with self.subTest(club=spellings[0]):
                seen = {teamcolors.colors(s)["primary"] for s in spellings}
                self.assertEqual(len(seen), 1, f"{spellings} disagree: {seen}")
                self.assertNotEqual(
                    seen.pop(), teamcolors.DEFAULT,
                    f"{spellings[0]} fell through to the default colour")

    def test_an_unknown_club_gets_the_neutral_default_not_a_guess(self):
        c = teamcolors.colors("Some Club That Does Not Exist")
        self.assertEqual(c["primary"], teamcolors.DEFAULT)

    def test_missing_and_empty_names_are_safe(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertEqual(teamcolors.colors(value)["primary"], teamcolors.DEFAULT)


class ReadabilityTests(unittest.TestCase):
    def test_white_shirted_clubs_get_dark_banner_text(self):
        for club in ("Real Madrid", "Fulham", "Valencia", "England"):
            with self.subTest(club=club):
                self.assertEqual(teamcolors.colors(club)["on_primary"], "#111418")

    def test_dark_shirted_clubs_get_light_banner_text(self):
        for club in ("Liverpool", "Chelsea", "Newcastle", "Tottenham"):
            with self.subTest(club=club):
                self.assertEqual(teamcolors.colors(club)["on_primary"], "#FFFFFF")

    def test_banner_text_always_contrasts_with_its_banner(self):
        """Every club must clear WCAG AA for large text (3:1).

        The banner carries the player's name at display size, so 3:1 is the
        applicable threshold rather than the 4.5:1 body-text one. Deriving the
        text colour rather than hand-pairing it means the real worst case is
        4.33:1, so this asserts 4.3 — a tighter bound catches a regression that
        a 3.0 assertion would sleep through.
        """
        for club in teamcolors.known_teams():
            with self.subTest(club=club):
                c = teamcolors.colors(club)
                self.assertGreaterEqual(
                    _contrast(c["primary"], c["on_primary"]), 4.3,
                    f"{club}: {c['on_primary']} on {c['primary']} is unreadable")

    def test_the_derived_text_colour_is_the_better_of_the_two(self):
        for club in teamcolors.known_teams():
            with self.subTest(club=club):
                primary = teamcolors.colors(club)["primary"]
                chosen = teamcolors.on_primary_for(primary)
                other = "#FFFFFF" if chosen == "#111418" else "#111418"
                self.assertGreaterEqual(
                    teamcolors.contrast_ratio(primary, chosen),
                    teamcolors.contrast_ratio(primary, other))

    def test_pale_primaries_get_a_darker_accent_for_dark_backgrounds(self):
        # A white accent on the dark player page would be invisible.
        self.assertNotEqual(teamcolors.colors("Real Madrid")["accent"], "#FFFFFF")
        # A saturated one is used as-is.
        self.assertEqual(teamcolors.colors("Liverpool")["accent"], "#C8102E")


class CoverageTests(unittest.TestCase):
    def test_every_current_season_club_has_a_real_colour(self):
        """A club on the pitch this season must not render with the fallback."""
        try:
            df = pl.read_parquet(config.SILVER / "results" / "results.parquet")
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"results lake unavailable: {exc}")

        current = df.filter(pl.col("season").is_in(["2026/27", "2026-27"]))
        missing: list[str] = []
        for code in ("E0", "SP1", "UCL"):
            sub = current.filter(pl.col("competition_code") == code)
            names = set(sub["home_team"].to_list()) | set(sub["away_team"].to_list())
            missing += [n for n in sorted(names)
                        if teamcolors.colors(n)["primary"] == teamcolors.DEFAULT]
        self.assertEqual(missing, [], f"clubs without a colour: {missing}")


def _contrast(a: str, b: str) -> float:
    la, lb = teamcolors._relative_luminance(a), teamcolors._relative_luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


if __name__ == "__main__":
    unittest.main()
