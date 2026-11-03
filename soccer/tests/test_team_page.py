"""The club page: profile assembly and the fixture de-duplication behind it."""
import unittest

from futball import clubs, teampage


def _skip_without_directory(test):
    if not clubs.directory():
        test.skipTest("club directory not cached; run `make refresh`")


class ProfileTests(unittest.TestCase):
    def setUp(self):
        _skip_without_directory(self)
        self.profile = teampage.profile("Barcelona")
        if not self.profile:
            self.skipTest("Barcelona not cached")

    def test_club_metadata_comes_through(self):
        self.assertEqual(self.profile["venue"], "Camp Nou")
        self.assertEqual(self.profile["founded"], 1899)
        self.assertEqual(self.profile["country"], "Spain")

    def test_colours_are_the_clubs_own(self):
        self.assertEqual(self.profile["colors"]["primary"], "#A50044")

    def test_the_coach_record_covers_only_the_current_manager(self):
        coach = self.profile["coach"]
        self.assertIsNotNone(coach)
        self.assertTrue(coach["name"])
        # The limitation is stated rather than a fabricated history shown.
        self.assertIn("current coach", coach["note"])

    def test_form_is_oldest_first_so_it_reads_left_to_right(self):
        dates = [m["date"] for m in self.profile["form"]]
        self.assertEqual(dates, sorted(dates))

    def test_the_squad_is_published(self):
        self.assertGreater(len(self.profile["squad"]), 15)

    def test_history_runs_newest_first(self):
        years = [h["season_year"] for h in self.profile["history"]]
        self.assertEqual(years, sorted(years, reverse=True))

    def test_limitations_name_what_is_missing(self):
        joined = " ".join(self.profile["limitations"]).lower()
        self.assertIn("capacity", joined)

    def test_an_unknown_club_returns_nothing(self):
        self.assertIsNone(teampage.profile("Not A Real Club"))


class NameResolutionTests(unittest.TestCase):
    """One club, one spelling — the page must agree with its own table."""

    def setUp(self):
        _skip_without_directory(self)

    def test_an_alias_resolves_to_the_lake_spelling(self):
        for alias, expected in (("Barça", "Barcelona"), ("Atleti", "Ath Madrid")):
            with self.subTest(alias=alias):
                p = teampage.profile(alias)
                self.assertIsNotNone(p, f"{alias} did not resolve")
                self.assertEqual(p["team"], expected)

    def test_the_heading_matches_the_name_used_for_colours(self):
        """Atlético's banner came out slate grey when these disagreed."""
        from futball import teamcolors
        for name in ("Barcelona", "Ath Madrid", "Arsenal"):
            with self.subTest(team=name):
                p = teampage.profile(name)
                self.assertEqual(p["colors"], teamcolors.colors(p["team"]))

    def test_the_heading_matches_the_standings_row(self):
        p = teampage.profile("Barça")
        self.assertEqual(p["team"], p["standing"]["team"])


class FixtureDeduplicationTests(unittest.TestCase):
    """Two sources listed one match on adjacent dates; it must appear once."""

    def setUp(self):
        _skip_without_directory(self)

    def test_a_pairing_is_not_listed_twice_within_a_day(self):
        for team in ("Barcelona", "Arsenal", "Real Madrid"):
            fixtures = teampage.fixtures(team, limit=20)
            seen: dict[tuple[str, str, str], list[str]] = {}
            for f in fixtures:
                key = (f["competition_code"], f["home_team"], f["away_team"])
                for previous in seen.get(key, []):
                    with self.subTest(team=team, pairing=key):
                        self.assertGreater(
                            abs(_days(f["date"], previous)), 1,
                            f"{key} listed twice within a day")
                seen.setdefault(key, []).append(f["date"])


class DifficultyTests(unittest.TestCase):
    def setUp(self):
        _skip_without_directory(self)

    def test_difficulty_is_bounded_and_labelled(self):
        d = teampage.fixture_difficulty("Barcelona", 3)
        for n in d["next"]:
            with self.subTest(opponent=n["opponent"]):
                self.assertGreaterEqual(n["difficulty"], 0.0)
                self.assertLessEqual(n["difficulty"], 1.0)
                self.assertTrue(n["label"])

    def test_it_says_it_is_a_guide_rather_than_a_projection(self):
        self.assertIn("guide", teampage.fixture_difficulty("Barcelona", 3)["note"])


def _days(a: str, b: str) -> int:
    from datetime import datetime
    return (datetime.strptime(a[:10], "%Y-%m-%d")
            - datetime.strptime(b[:10], "%Y-%m-%d")).days


if __name__ == "__main__":
    unittest.main()


class SeasonStatsTests(unittest.TestCase):
    """The panel that replaced the un-buildable "last starting XI"."""

    def setUp(self):
        _skip_without_directory(self)
        self.stats = teampage.profile("Barcelona", "SP1")["season_stats"]

    def test_the_domestic_season_fills_every_metric(self):
        from futball import teamstats
        self.assertEqual(len(self.stats["metrics"]), len(teamstats.METRICS))

    def test_each_metric_is_ranked_within_the_division(self):
        for m in self.stats["metrics"]:
            with self.subTest(metric=m["key"]):
                self.assertGreaterEqual(m["rank"], 1)
                self.assertLessEqual(m["rank"], m["of"])

    def test_rank_direction_follows_the_metric(self):
        """Fewest fouls ranks first, not most.

        Compared by value rather than by club: clubs tie on these rates, and a
        shared rank cannot pick between them.
        """
        from futball import teamstats
        fouls = next(m for m in self.stats["metrics"] if m["key"] == "fouls_per_match")
        board = teamstats.leaders("SP1", "fouls_per_match", limit=50)
        self.assertEqual(board["rows"][fouls["rank"] - 1]["value"], fouls["value"])
        # And the direction itself: the board must run low-to-high for fouls.
        values = [r["value"] for r in board["rows"]]
        self.assertEqual(values, sorted(values))

    def test_a_competition_without_the_club_says_so(self):
        from futball import teamstats
        empty = teamstats.club("Barcelona", "E0")
        self.assertEqual(empty["metrics"], [])
        self.assertIn("no matches", empty["note"])

    def test_lineup_source_and_unavailable_ratings_are_stated(self):
        joined = " ".join(teampage.profile("Barcelona")["limitations"]).lower()
        self.assertIn("espn", joined)
        self.assertIn("match ratings are not supplied", joined)


class SeasonXiTests(unittest.TestCase):
    """The pitch panel: real contributions only, no guessed line-up."""

    def setUp(self):
        _skip_without_directory(self)
        from futball import seasonxi
        self.seasonxi = seasonxi
        self.xi = seasonxi.season_xi("Barcelona", "SP1")

    def test_the_pitch_has_eleven_positions(self):
        self.assertEqual(len(self.xi["slots"]), 11)

    def test_only_players_with_a_contribution_are_placed(self):
        for slot in self.xi["slots"]:
            p = slot["player"]
            if p:
                with self.subTest(player=p["name"]):
                    self.assertGreater(p["goals"] + p["assists"], 0)

    def test_players_land_on_their_own_line(self):
        """A forward must not be filed into defence to fill the picture."""
        from futball import clubs
        squad = {m["player_id"]: m["position"] for m in clubs.find("Barcelona")["squad"]}
        sections = {"Attack": "Offence", "Midfield": "Midfield",
                    "Defence": "Defence", "Goalkeeper": "Goalkeeper"}
        for slot in self.xi["slots"]:
            if slot["player"]:
                with self.subTest(slot=slot["slot"]):
                    self.assertEqual(squad[slot["player"]["player_id"]],
                                     sections[slot["line"]])

    def test_a_slot_with_nobody_stays_empty(self):
        self.assertLess(self.xi["filled"], 11)
        self.assertTrue(any(s["player"] is None for s in self.xi["slots"]))

    def test_ratings_sit_above_the_baseline_and_below_the_cap(self):
        for slot in self.xi["slots"]:
            p = slot["player"]
            if p and p["rating"] is not None:
                with self.subTest(player=p["name"]):
                    self.assertGreater(p["rating"], self.seasonxi.BASELINE)
                    self.assertLessEqual(
                        p["rating"],
                        self.seasonxi.BASELINE + self.seasonxi.CONTRIBUTION_CAP,
                    )

    def test_no_contribution_means_no_rating(self):
        self.assertIsNone(self.seasonxi.rate(goals=0, assists=0, played=10))

    def test_one_appearance_is_too_few_to_rate(self):
        self.assertIsNone(self.seasonxi.rate(goals=2, assists=0, played=1))

    def test_a_goal_counts_for_more_than_an_assist(self):
        self.assertGreater(self.seasonxi.rate(2, 0, 4), self.seasonxi.rate(0, 2, 4))

    def test_the_note_says_the_rating_is_ours(self):
        self.assertIn("own model", self.xi["note"])

    def test_the_note_explains_the_missing_lineup(self):
        self.assertIn("line-up", self.xi["note"].lower())

    def test_an_unknown_club_yields_nothing(self):
        self.assertEqual(self.seasonxi.contributors("Not A Club", "SP1"), [])


class KeeperAttributionTests(unittest.TestCase):
    """Clean sheets are the team's; the name beside them is the club's shirt 1."""

    def setUp(self):
        _skip_without_directory(self)

    def test_a_settled_number_one_is_resolved(self):
        for team, expected in (("Barcelona", "Joan García"),
                               ("Real Madrid", "Thibaut Courtois"),
                               ("Arsenal", "David Raya")):
            with self.subTest(team=team):
                keeper = clubs.first_choice_keeper(team)
                self.assertIsNotNone(keeper, f"{team} did not resolve")
                self.assertEqual(keeper["name"], expected)
                self.assertEqual(keeper["shirt"], 1)

    def test_two_keepers_wearing_one_resolves_to_neither(self):
        """Ambiguity is reported as unknown, never broken by picking first."""
        entry = clubs.find("Celta")
        if not entry:
            self.skipTest("Celta not cached")
        wearing_one = [
            p for p in entry.get("squad") or []
            if p.get("position") == "Goalkeeper"
            and clubs.squad_detail(p.get("name")).get("number") == 1
        ]
        if len(wearing_one) < 2:
            self.skipTest("Celta's squad no longer has the duplicate shirt")
        self.assertIsNone(clubs.first_choice_keeper("Celta"))

    def test_an_unknown_club_resolves_to_nothing(self):
        self.assertIsNone(clubs.first_choice_keeper("Not A Club"))
        self.assertIsNone(clubs.first_choice_keeper(None))

    def test_a_full_legal_name_still_finds_its_shirt(self):
        """"Álvaro Vallés Rosa" has to reach the squad's "Álvaro Vallés"."""
        self.assertEqual(clubs.squad_detail("Álvaro Vallés Rosa").get("number"), 1)

    def test_the_board_keeps_the_club_row_when_the_keeper_is_unknown(self):
        from futball import league
        from futball.api import app as api_app
        rows = api_app._with_keepers(league.clean_sheets("SP1")["rows"])
        self.assertTrue(rows)
        for r in rows:
            with self.subTest(team=r["team"]):
                self.assertIn("keeper", r)
                self.assertIsNotNone(r["team"])
                self.assertIsInstance(r["clean_sheets"], int)

    def test_the_note_does_not_claim_the_keeper_played(self):
        from futball.api import app as api_app
        leaders = api_app.get_player_leaders("SP1", limit=5)["data"]
        note = leaders["clean_sheets_note"].lower()
        self.assertIn("shirt rather than the team sheet", note)
        self.assertIn("team's", note)
