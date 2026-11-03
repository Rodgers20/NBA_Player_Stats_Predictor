"""Portrait resolution: which provider wins, and how names are matched.

The failure this guards against is putting the **wrong face** on a player. That
is worse than showing no face, because it silently misinforms and looks
deliberate, so every matching rule here errs toward returning nothing.
"""
import unittest
from unittest.mock import patch

from futball import headshots


SQUAD = [
    {"name": "B. Saka", "photo": "https://cdn/1460.png", "player_id": 1460},
    {"name": "Lamine Yamal", "photo": "https://cdn/386828.png", "player_id": 386828},
    {"name": "K. Mbappé", "photo": "https://cdn/278.png", "player_id": 278},
    {"name": "M. Ødegaard", "photo": "https://cdn/37127.png", "player_id": 37127},
    # Two different players who reduce to the same initial+surname key.
    {"name": "J. Sánchez", "photo": "https://cdn/aaa.png", "player_id": 1},
    {"name": "J. Sanchez", "photo": "https://cdn/bbb.png", "player_id": 2},
    {"name": "Nomatch", "photo": None, "player_id": 3},
]


class IndexTests(unittest.TestCase):
    def setUp(self):
        p = patch("futball.sources.api_football.cached_squads", return_value=SQUAD)
        p.start()
        self.addCleanup(p.stop)
        headshots.refresh_caches()
        self.addCleanup(headshots.refresh_caches)

    def test_an_exact_name_resolves(self):
        self.assertEqual(headshots.portrait_url("Lamine Yamal"), "https://cdn/386828.png")

    def test_a_full_legal_name_resolves_by_leading_run(self):
        # StatsBomb spells him "Lamine Yamal Nasraoui Ebana".
        self.assertEqual(
            headshots.portrait_url("Lamine Yamal Nasraoui Ebana"),
            "https://cdn/386828.png")

    def test_an_abbreviated_squad_name_resolves_from_the_full_name(self):
        """68% of squad entries are abbreviated, so this is the common path."""
        self.assertEqual(headshots.portrait_url("Bukayo Saka"), "https://cdn/1460.png")
        self.assertEqual(headshots.portrait_url("Martin Ødegaard"), "https://cdn/37127.png")

    def test_accents_do_not_prevent_a_match(self):
        self.assertEqual(headshots.portrait_url("Kylian Mbappe"), "https://cdn/278.png")
        self.assertEqual(headshots.portrait_url("Kylian Mbappé Lottin"), "https://cdn/278.png")

    def test_an_ambiguous_initial_and_surname_resolves_to_nothing(self):
        """Two J. Sanchez entries must yield no portrait rather than a coin flip."""
        self.assertIsNone(headshots.portrait_url("Jorge Sanchez"))
        self.assertIsNone(headshots.portrait_url("Javier Sánchez"))

    def test_a_surname_alone_is_not_enough(self):
        # "Saka" could be any number of people; one token never matches.
        self.assertIsNone(headshots.portrait_url("Saka"))

    def test_an_unknown_player_resolves_to_nothing(self):
        self.assertIsNone(headshots.portrait_url("Not A Real Person"))

    def test_a_squad_row_without_a_photo_is_not_indexed(self):
        self.assertIsNone(headshots.portrait_url("Nomatch Someone"))

    def test_empty_input_is_safe(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertIsNone(headshots.portrait_url(value))


class InitialKeyTests(unittest.TestCase):
    def test_initial_key_matches_the_feed_format(self):
        self.assertEqual(headshots._initial_key("Bukayo Saka"), "b saka")
        self.assertEqual(headshots._initial_key("B. Saka"), "b saka")

    def test_a_single_token_has_no_initial_key(self):
        self.assertIsNone(headshots._initial_key("Raphinha"))
        self.assertIsNone(headshots._initial_key(""))


class TeamMatchTests(unittest.TestCase):
    """Directory clubs must reach API-Football's differently-spelled teams."""

    TEAM_IDS = {
        "Arsenal": 42,
        "Atletico Madrid": 530,
        "Barcelona": 529,
        "Real Madrid": 541,
    }

    def _match(self, club, short=None):
        entry = {"name": club, "short_name": short}
        return headshots._match_team_id(club, entry, self.TEAM_IDS)

    def test_the_short_name_is_used_when_the_long_one_differs(self):
        self.assertEqual(self._match("Arsenal FC", "Arsenal"), 42)
        self.assertEqual(self._match("FC Barcelona", "Barça"), 529)

    def test_normalised_comparison_bridges_accents(self):
        self.assertEqual(self._match("Club Atlético de Madrid", "Atleti"), 530)

    def test_an_unknown_club_returns_none(self):
        self.assertIsNone(self._match("Some Club That Does Not Exist", "Nope"))


class ProviderOrderTests(unittest.TestCase):
    """Only provider portraits are allowed; misses use frontend initials."""

    def setUp(self):
        p = patch("futball.sources.api_football.cached_squads", return_value=SQUAD)
        p.start()
        self.addCleanup(p.stop)
        headshots.refresh_caches()
        self.addCleanup(headshots.refresh_caches)

        # Isolate the manifest so the real cache is neither read nor written.
        manifest = patch.object(headshots, "_manifest", {})
        manifest.start()
        self.addCleanup(manifest.stop)
        save = patch.object(headshots, "_save")
        save.start()
        self.addCleanup(save.stop)

    def test_a_squad_portrait_wins_and_is_recorded_as_such(self):
        with patch.object(headshots, "_download", return_value="abc.png") as dl:
            h = headshots.resolve("Bukayo Saka")
        self.assertTrue(h.found)
        self.assertEqual(h.provider, "api-football")
        self.assertEqual(h.image_url, "https://cdn/1460.png")
        dl.assert_called_once()

    def test_missing_portrait_does_not_contact_wikipedia(self):
        with patch.object(headshots, "_client") as client:
            h = headshots.resolve("Some Unlisted Player")
        self.assertFalse(h.found)
        self.assertIsNone(h.file)
        client.assert_not_called()

    def test_failed_download_is_retried_without_a_wikipedia_fallback(self):
        with patch.object(headshots, "_download", return_value=None) as dl:
            h = headshots.resolve("Bukayo Saka")
            again = headshots.resolve("Bukayo Saka")
        self.assertFalse(h.found)
        self.assertIsNone(h.file)
        self.assertEqual(h.provider, "api-football")
        self.assertFalse(again.found)
        dl.assert_called_once()

    def test_legacy_cache_is_not_served_and_gets_replaced(self):
        import tempfile
        from pathlib import Path
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as d, patch.object(headshots, "CACHE_DIR", Path(d)):
            Path(d, "legacy.jpg").write_bytes(b"old photo")
            for provider in ("wikipedia", None):
                headshots._manifest["bukayo saka"] = asdict(headshots.Headshot(
                    name="Bukayo Saka", found=True, provider=provider, file="legacy.jpg"))
                self.assertIsNone(headshots.cached_path("Bukayo Saka"))
                with patch.object(headshots, "_download", return_value="new.png"):
                    h = headshots.resolve("Bukayo Saka")
                self.assertEqual(h.provider, "api-football")
                self.assertEqual(h.file, "new.png")
                self.assertTrue(Path(d, "legacy.jpg").exists())

    def test_newly_available_portrait_bypasses_cached_miss(self):
        with patch.object(headshots, "portrait_url", return_value=None):
            self.assertFalse(headshots.resolve("Bukayo Saka").found)
        with patch.object(headshots, "_download", return_value="new.png"):
            self.assertTrue(headshots.resolve("Bukayo Saka").found)


if __name__ == "__main__":
    unittest.main()
