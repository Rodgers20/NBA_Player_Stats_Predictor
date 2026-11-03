"""Calendar feeds, news parsing, and the project's own player rating."""
import unittest
from datetime import datetime, timezone

from futball import calendar_feed, ratings
from futball.sources import news


class CalendarTests(unittest.TestCase):
    NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)

    FIXTURES = [
        {
            "competition_code": "UCL", "fixture_key": "UCL|2026-10-13|Arsenal|Lille",
            "kickoff_utc": "2026-10-13T19:00:00+00:00",
            "home_team": "Arsenal", "away_team": "Lille",
            "round": "Matchday 2", "referee": None,
        },
    ]

    def test_a_document_is_well_formed(self):
        ics = calendar_feed._document("Test", calendar_feed._events(self.FIXTURES, self.NOW))
        self.assertTrue(ics.startswith("BEGIN:VCALENDAR\r\n"))
        self.assertTrue(ics.rstrip().endswith("END:VCALENDAR"))
        self.assertEqual(ics.count("BEGIN:VEVENT"), ics.count("END:VEVENT"))

    def test_lines_end_with_crlf_as_the_spec_requires(self):
        ics = calendar_feed._document("Test", calendar_feed._events(self.FIXTURES, self.NOW))
        # A bare LF anywhere would break stricter clients.
        self.assertNotIn("\n", ics.replace("\r\n", ""))

    def test_the_uid_is_stable_for_the_same_fixture(self):
        """The whole point: a re-subscribe must update, not duplicate."""
        first = calendar_feed._events(self.FIXTURES, self.NOW)
        later = calendar_feed._events(self.FIXTURES, datetime(2026, 12, 1, tzinfo=timezone.utc))
        uid_of = lambda ls: next(l for l in ls if l.startswith("UID:"))  # noqa: E731
        self.assertEqual(uid_of(first), uid_of(later))

    def test_the_sequence_changes_when_the_kickoff_moves(self):
        moved = [{**self.FIXTURES[0], "kickoff_utc": "2026-10-13T20:00:00+00:00"}]
        seq = lambda fx: next(  # noqa: E731
            l for l in calendar_feed._events(fx, self.NOW) if l.startswith("SEQUENCE:"))
        self.assertNotEqual(seq(self.FIXTURES), seq(moved))

    def test_the_event_lasts_a_realistic_match(self):
        lines = calendar_feed._events(self.FIXTURES, self.NOW)
        start = next(l for l in lines if l.startswith("DTSTART:"))
        end = next(l for l in lines if l.startswith("DTEND:"))
        self.assertEqual(start, "DTSTART:20261013T190000Z")
        self.assertEqual(end, "DTEND:20261013T205500Z")  # 115 minutes

    def test_special_characters_are_escaped(self):
        odd = [{**self.FIXTURES[0], "home_team": "A; B, C", "away_team": "D"}]
        lines = calendar_feed._events(odd, self.NOW)
        summary = next(l for l in lines if l.startswith("SUMMARY:"))
        self.assertIn("\\;", summary)
        self.assertIn("\\,", summary)

    def test_a_fixture_without_a_kickoff_is_skipped(self):
        self.assertEqual(
            calendar_feed._events([{**self.FIXTURES[0], "kickoff_utc": None}], self.NOW), [])


class NewsParsingTests(unittest.TestCase):
    RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item>
        <title>Arsenal win late at Anfield</title>
        <link>https://example.com/a</link>
        <pubDate>Sat, 13 Sep 2026 18:30:00 GMT</pubDate>
        <description>A &lt;b&gt;late&lt;/b&gt; winner.</description>
      </item>
      <item><title>No link here</title></item>
    </channel></rss>"""

    def test_items_parse_to_headline_source_and_link(self):
        items = news.parse_feed(self.RSS, news.FEEDS[0])
        self.assertEqual(len(items), 1, "an item without a link must be dropped")
        item = items[0]
        self.assertEqual(item["title"], "Arsenal win late at Anfield")
        self.assertEqual(item["link"], "https://example.com/a")
        self.assertEqual(item["source"], "BBC Sport")

    def test_markup_is_stripped_from_the_summary(self):
        item = news.parse_feed(self.RSS, news.FEEDS[0])[0]
        self.assertEqual(item["summary"], "A late winner.")
        self.assertNotIn("<", item["summary"])

    def test_the_publish_time_becomes_utc_iso(self):
        item = news.parse_feed(self.RSS, news.FEEDS[0])[0]
        self.assertTrue(item["published"].startswith("2026-09-13T18:30:00+00:00"))

    def test_malformed_xml_yields_nothing_rather_than_raising(self):
        self.assertEqual(news.parse_feed("<rss><channel>", news.FEEDS[0]), [])


class RatingTests(unittest.TestCase):
    def test_an_average_contribution_sits_near_the_baseline(self):
        value, _ = ratings.rate({"goals": 0, "assists": 0, "xg": 0.0}, 90)
        self.assertEqual(value, ratings.BASELINE)

    def test_goals_raise_the_rating(self):
        quiet, _ = ratings.rate({"goals": 0, "xg": 0.0}, 90)
        scorer, _ = ratings.rate({"goals": 2, "xg": 0.8}, 90)
        self.assertGreater(scorer, quiet)

    def test_losing_the_ball_lowers_it(self):
        clean, _ = ratings.rate({"goals": 0, "xg": 0.0}, 90)
        sloppy, _ = ratings.rate({"goals": 0, "xg": 0.0, "dispossessed": 8,
                                  "miscontrols": 6}, 90)
        self.assertLess(sloppy, clean)

    def test_no_single_component_can_carry_the_rating(self):
        """A ten-goal outlier must not paper over everything else."""
        _value, components = ratings.rate({"goals": 10, "xg": 1.0}, 90)
        for key, contribution in components.items():
            with self.subTest(component=key):
                self.assertLessEqual(abs(contribution), ratings.COMPONENT_CAP)

    def test_the_rating_stays_inside_its_range(self):
        for stats in ({"goals": 40, "assists": 20, "xg": 0.0},
                      {"dispossessed": 200, "miscontrols": 200, "fouls_committed": 100}):
            with self.subTest(stats=stats):
                value, _ = ratings.rate(stats, 90)
                self.assertGreaterEqual(value, ratings.FLOOR)
                self.assertLessEqual(value, ratings.CEILING)

    def test_a_short_cameo_is_not_projected_to_a_full_match(self):
        """Ten minutes and a goal must not become a per-90 masterpiece."""
        ten_min, _ = ratings.rate({"goals": 1, "xg": 0.3}, 10)
        self.assertLessEqual(ten_min, ratings.CEILING)
        # And the leaderboard excludes them entirely.
        self.assertGreater(ratings.MIN_MINUTES, 10)

    def test_the_model_publishes_its_own_weights(self):
        described = ratings.describe()
        self.assertEqual(described["name"], "FUTBALL rating")
        self.assertIn("weights", described)
        self.assertIn("not", described["note"].lower())
        self.assertIn("fotmob", described["note"].lower())


class TeamOfTheWeekTests(unittest.TestCase):
    """An XI belongs to one competition.

    Pooling them put Euro 2024 internationals on the Champions League page,
    which is a claim about the wrong league entirely.
    """

    def test_a_competition_without_event_data_returns_an_empty_xi_and_says_why(self):
        for code in ("E0", "SP1", "UCL"):
            with self.subTest(code=code):
                totw = ratings.team_of_the_week(code)
                self.assertEqual(totw["players"], [])
                self.assertEqual(totw["competition_code"], code)
                self.assertIn("per-player match data", totw["note"])

    def test_an_unknown_competition_does_not_raise(self):
        totw = ratings.team_of_the_week("NOPE")
        self.assertEqual(totw["players"], [])

    def test_a_competition_with_event_data_fills_the_shape(self):
        """Exercised through the real builder by mapping a code to the lake."""
        from unittest.mock import patch
        from futball import config, marts
        try:
            available = marts.competitions()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"event lake unavailable: {exc}")
        if not available:
            self.skipTest("no event competitions")

        competition_id = available[0]["competition_id"]
        with patch.dict(config.STATSBOMB_COMPETITION_IDS, {"UCL": competition_id}):
            totw = ratings.team_of_the_week("UCL")

        self.assertEqual(len(totw["players"]), 11)
        picked = [s["player"]["player_id"] for s in totw["players"] if s["player"]]
        self.assertEqual(len(picked), len(set(picked)), "a player appears twice")

        roles = {slot: set(r) for slot, r, _x, _y in ratings.FORMATION}
        for s in totw["players"]:
            if s["player"]:
                self.assertIn(s["player"]["position"], roles[s["slot"]])

    def test_each_pick_carries_a_surname_and_a_club_for_the_graphic(self):
        from unittest.mock import patch
        from futball import config, marts
        try:
            available = marts.competitions()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"event lake unavailable: {exc}")
        if not available:
            self.skipTest("no event competitions")

        with patch.dict(config.STATSBOMB_COMPETITION_IDS,
                        {"UCL": available[0]["competition_id"]}):
            totw = ratings.team_of_the_week("UCL")
        for s in totw["players"]:
            if not s["player"]:
                continue
            with self.subTest(slot=s["slot"]):
                self.assertTrue(s["player"]["short_name"])
                self.assertNotIn(" ", s["player"]["short_name"].strip().rstrip("."),
                                 "a surname should be one word unless it has a particle")


class SurnameTests(unittest.TestCase):
    def test_spanish_names_use_the_first_surname(self):
        """"Marc Cucurella Saseta" is Cucurella; the last token is maternal."""
        for full, expected in [
            ("Marc Cucurella Saseta", "Cucurella"),
            ("Daniel Olmo Carvajal", "Olmo"),
            ("Nicholas Williams Arthuer", "Williams"),
            ("Unai Simón Mendibil", "Simón"),
        ]:
            with self.subTest(name=full):
                self.assertEqual(ratings.familiar_surname(full), expected)

    def test_a_particle_stays_attached(self):
        self.assertEqual(ratings.familiar_surname("Virgil van Dijk"), "van Dijk")

    def test_two_token_names_use_the_last_token(self):
        self.assertEqual(ratings.familiar_surname("Cole Palmer"), "Palmer")

    def test_a_single_name_is_left_alone(self):
        self.assertEqual(ratings.familiar_surname("Raphinha"), "Raphinha")

    def test_empty_input_is_safe(self):
        self.assertEqual(ratings.familiar_surname(None), "")
        self.assertEqual(ratings.familiar_surname(""), "")


class NewsClubMatchingTests(unittest.TestCase):
    """Word boundaries, because a substring test claimed "Inter Miami" for Inter."""

    def setUp(self):
        from futball import clubs
        if not clubs.directory():
            self.skipTest("club directory not cached")

    def _clubs_for(self, headline: str) -> set[str]:
        from unittest.mock import patch
        item = {"title": headline, "link": "x", "source": "T", "source_key": "t",
                "published": None, "summary": None}
        with patch.object(news, "headlines", return_value=[item]):
            return {i["club"] for i in news.for_competition("UCL", limit=5)["matched"]}

    def test_a_different_club_sharing_the_name_is_not_a_match(self):
        """"Inter Miami" contains the whole word "Inter" — boundaries cannot
        separate these, so the exclusion list has to."""
        self.assertNotIn("Inter", self._clubs_for("Messi and Inter Miami win again"))

    def test_a_club_name_inside_an_ordinary_word_is_not_a_match(self):
        for headline in ("An international friendly", "Read the full interview"):
            with self.subTest(headline=headline):
                self.assertNotIn("Inter", self._clubs_for(headline))

    def test_a_real_mention_still_matches(self):
        self.assertIn("Inter", self._clubs_for("Inter beat Milan in the derby"))
