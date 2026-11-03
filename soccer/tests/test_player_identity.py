"""One person, one search result — and always the club's colours.

Two bugs these lock down, both reported from the running app:

1. Searching "Lamine" returned three rows — his Euro 2024 spell and one per
   league — because the sources were never reconciled into one identity.
2. His Euro 2024 page was coloured Spain red and Mbappé's France blue, because
   the StatsBomb rows know only the national team. Colour must always come from
   the club.
"""
import unittest

from futball import clubs, players, teamcolors


def _skip_without_directory(test):
    if not clubs.directory():
        test.skipTest("club directory not cached; run `make refresh`")


class MatchKeyTests(unittest.TestCase):
    def test_full_legal_name_and_common_name_share_a_key(self):
        pairs = [
            ("Lamine Yamal Nasraoui Ebana", "Lamine Yamal"),
            ("Kylian Mbappé Lottin", "Kylian Mbappé"),
            ("Robert Lewandowski", "Robert Lewandowski"),
        ]
        for full, common in pairs:
            with self.subTest(player=common):
                self.assertEqual(players._match_key(full), players._match_key(common))

    def test_different_players_do_not_collide(self):
        self.assertNotEqual(
            players._match_key("Kylian Mbappé"), players._match_key("Ethan Mbappé"))

    def test_a_single_token_name_still_produces_a_key(self):
        self.assertEqual(players._match_key("Raphinha"), "raphinha")

    def test_accents_and_punctuation_do_not_split_an_identity(self):
        self.assertEqual(
            players._match_key("Kylian Mbappé"), players._match_key("Kylian Mbappe"))


class SearchMergeTests(unittest.TestCase):
    def setUp(self):
        _skip_without_directory(self)
        players.refresh_caches()

    def test_a_player_in_three_competitions_returns_one_row(self):
        hits = players.search("Lamine")
        self.assertEqual(len(hits), 1, f"expected one Lamine, got {[h['name'] for h in hits]}")
        self.assertGreaterEqual(len(hits[0]["competitions"]), 2,
                                "the competitions should be listed on the single row")

    def test_the_row_carries_the_club_not_the_country(self):
        hit = players.search("Mbapp")[0]
        self.assertEqual(hit["club"], "Real Madrid")
        self.assertNotEqual(hit["club"], "France")

    def test_a_short_query_returns_nothing(self):
        self.assertEqual(players.search("a"), [])
        self.assertEqual(players.search(""), [])


class ClubColourTests(unittest.TestCase):
    """Colours come from the club under every source."""

    def setUp(self):
        _skip_without_directory(self)
        players.refresh_caches()

    def test_a_euro_2024_page_uses_club_colours_not_national_ones(self):
        profile = players.profile(players.search("Mbapp")[0]["player_id"])
        self.assertEqual(profile["club"], "Real Madrid")
        # Real Madrid white, never France's #21304F.
        self.assertEqual(profile["colors"]["primary"], "#FFFFFF")
        self.assertNotEqual(profile["colors"]["primary"],
                            teamcolors.colors("France")["primary"])

    def test_barcelona_colours_for_yamal(self):
        profile = players.profile(players.search("Lamine")[0]["player_id"])
        self.assertEqual(profile["colors"]["primary"], "#A50044")

    def test_colour_does_not_change_between_spells(self):
        """Switching competition must not repaint the page."""
        profile = players.profile(players.search("Lamine")[0]["player_id"])
        self.assertGreater(len(profile["spells"]), 1)
        # Colour lives on the identity, not the spell — assert it is not there.
        for spell in profile["spells"]:
            self.assertNotIn("colors", spell)

    def test_an_unknown_club_falls_back_to_neutral_never_a_country(self):
        self.assertEqual(teamcolors.colors(None)["primary"], teamcolors.DEFAULT)


class ProfileShapeTests(unittest.TestCase):
    def setUp(self):
        _skip_without_directory(self)
        players.refresh_caches()

    def test_spells_run_domestic_league_then_ucl_then_international(self):
        """Reading order, not data-richness order.

        The Euro spell carries far more detail, but someone looking a player up
        wants the competition they actually play in week to week, so the page
        leads with their league.
        """
        profile = players.profile(players.search("Lamine")[0]["player_id"])
        order = [s["competition_code"] for s in profile["spells"]]
        rank = [players._COMPETITION_RANK.get(c, players._UNRANKED) for c in order]
        self.assertEqual(rank, sorted(rank), f"out of order: {order}")
        self.assertIn(order[0], ("E0", "SP1"), "a domestic league must come first")
        self.assertEqual(order[-1], "EURO2024", "internationals come last")

    def test_the_page_opens_on_the_first_spell(self):
        profile = players.profile(players.search("Lamine")[0]["player_id"])
        self.assertEqual(profile["default_spell"], 0)
        opened = profile["spells"][0]["competition_code"]
        self.assertIn(opened, ("E0", "SP1"))

    def test_every_spell_declares_the_sections_it_can_fill(self):
        profile = players.profile(players.search("Lamine")[0]["player_id"])
        for spell in profile["spells"]:
            with self.subTest(competition=spell["competition"]):
                self.assertTrue(spell["sections"])
                if spell["detail"] == "basic":
                    self.assertNotIn("shot_map", spell["sections"])
                    self.assertNotIn("heatmap", spell["sections"])

    def test_the_current_listing_beats_the_two_year_old_specific_role(self):
        """Current before specific — the reverse of what this once asserted.

        StatsBomb names the exact role, but its data here is Euro 2024. It
        described Dembélé as a Right Winger long after he moved to striker, so
        the coarse-but-current squad listing wins the headline and the granular
        role is shown as a secondary instead.
        """
        profile = players.profile(players.search("Lamine")[0]["player_id"])
        self.assertEqual(profile["position"], "Offence")
        detail = profile["position_detail"]
        self.assertEqual(detail["primary"]["code"], "FW")
        self.assertIn("RW", [s["code"] for s in detail["secondary"]])

    def test_legacy_ids_still_resolve(self):
        """Old links used a bare event id or an `fd-` id; both must still work."""
        canonical = players.search("Lamine")[0]["player_id"]
        by_event = players.profile("316046")
        self.assertIsNotNone(by_event)
        self.assertEqual(by_event["player_id"], canonical)

    def test_an_unknown_id_returns_none_rather_than_raising(self):
        self.assertIsNone(players.profile("not-a-player"))
        self.assertIsNone(players.profile("fd-99999999"))


class ClubDirectoryTests(unittest.TestCase):
    def setUp(self):
        _skip_without_directory(self)

    def test_short_and_long_spellings_reach_one_crest(self):
        for spellings in [
            ("Barça", "Barcelona", "FC Barcelona"),
            ("Celta", "Celta Vigo", "RC Celta de Vigo"),
            ("Real Madrid", "Real Madrid CF"),
        ]:
            with self.subTest(club=spellings[0]):
                crests = {clubs.crest(s) for s in spellings}
                self.assertEqual(len(crests), 1, f"{spellings} disagree: {crests}")
                self.assertIsNotNone(crests.pop())

    def test_the_clubs_that_rendered_as_initials_now_have_crests(self):
        # "Barça" drew "BA" and "Celta" drew "C" before the directory existed.
        for name in ("Barça", "Celta", "Nott'm Forest", "La Coruna", "Espanol"):
            with self.subTest(club=name):
                self.assertIsNotNone(clubs.crest(name), f"{name} has no crest")

    def test_squad_lookup_finds_a_club_for_a_full_legal_name(self):
        self.assertIsNotNone(clubs.club_of("Kylian Mbappé Lottin"))
        self.assertIsNotNone(clubs.club_of("Lamine Yamal Nasraoui Ebana"))

    def test_an_unknown_player_has_no_club(self):
        self.assertIsNone(clubs.club_of("Not A Real Person"))


if __name__ == "__main__":
    unittest.main()
