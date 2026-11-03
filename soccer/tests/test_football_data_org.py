"""football-data.org adapter.

The risk this module carries is not parsing — it is *duplication*. The same
Premier League match arrives from football-data.co.uk as
"Nott'm Forest v Tottenham" and from football-data.org as
"Nottingham Forest FC v Tottenham Hotspur FC". If both reach the results lake
unreconciled the league table counts that match twice, which is very hard to
spot and completely wrong. Most of what follows guards that boundary.
"""
import json
import time
import unittest
from unittest.mock import patch

from futball.sources import football_data_org as fdo


def _match(home, away, hg, ag, *, status="FINISHED", utc="2026-09-12T14:00:00Z",
           hth=0, hta=0, season_start="2026-08-15", matchday=4):
    return {
        "utcDate": utc,
        "status": status,
        "matchday": matchday,
        "stage": "REGULAR_SEASON",
        "season": {"startDate": season_start, "endDate": "2027-05-23"},
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "score": {
            "winner": "HOME",
            "duration": "REGULAR",
            "fullTime": {"home": hg, "away": ag},
            "halfTime": {"home": hth, "away": hta},
        },
    }


def _payload(*matches):
    return {"count": len(matches), "matches": list(matches)}


class SeasonLabelTests(unittest.TestCase):
    def test_domestic_league_uses_slash_form(self):
        # results.py writes "2026/27" for football-data.co.uk divisions.
        self.assertEqual(fdo.season_label("E0", 2026), "2026/27")
        self.assertEqual(fdo.season_label("SP1", 2019), "2019/20")

    def test_champions_league_uses_hyphen_form(self):
        # openfootball CL files are named 2025-26, and results.py passes that
        # stem straight through as the season. Ours must agree or the same
        # season splits in two.
        self.assertEqual(fdo.season_label("UCL", 2026), "2026-27")

    def test_century_rollover(self):
        self.assertEqual(fdo.season_label("E0", 2099), "2099/00")


class ParseTests(unittest.TestCase):
    def test_finished_match_maps_to_a_result_row(self):
        rows = fdo.parse_matches(
            _payload(_match("Arsenal FC", "Chelsea FC", 2, 1, hth=1, hta=0)), "E0")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["date"], "2026-09-12")
        self.assertEqual(r["time"], "14:00")
        self.assertEqual(r["home_team"], "Arsenal FC")
        self.assertEqual(r["away_team"], "Chelsea FC")
        self.assertEqual((r["home_goals"], r["away_goals"]), (2, 1))
        self.assertEqual((r["ht_home_goals"], r["ht_away_goals"]), (1, 0))
        self.assertEqual(r["season"], "2026/27")
        self.assertEqual(r["matchday"], 4)

    def test_unplayed_and_abandoned_matches_are_skipped(self):
        for status in ("TIMED", "SCHEDULED", "IN_PLAY", "PAUSED",
                       "POSTPONED", "CANCELLED", "SUSPENDED"):
            with self.subTest(status=status):
                rows = fdo.parse_matches(
                    _payload(_match("A FC", "B FC", 1, 0, status=status)), "E0")
                self.assertEqual(rows, [], f"{status} must not produce a result")

    def test_awarded_match_counts_as_played(self):
        rows = fdo.parse_matches(
            _payload(_match("A FC", "B FC", 3, 0, status="AWARDED")), "E0")
        self.assertEqual(len(rows), 1)

    def test_finished_match_with_null_score_is_skipped(self):
        bad = _match("A FC", "B FC", None, None)
        self.assertEqual(fdo.parse_matches(_payload(bad), "E0"), [])

    def test_missing_halftime_score_is_none_not_zero(self):
        m = _match("A FC", "B FC", 1, 0)
        m["score"]["halfTime"] = {"home": None, "away": None}
        r = fdo.parse_matches(_payload(m), "E0")[0]
        self.assertIsNone(r["ht_home_goals"])
        self.assertIsNone(r["ht_away_goals"])

    def test_penalty_shootout_reports_the_regular_time_score(self):
        # A CL knockout decided on penalties is a draw in the table.
        m = _match("A FC", "B FC", 1, 1, status="FINISHED")
        m["score"]["duration"] = "PENALTY"
        m["score"]["penalties"] = {"home": 4, "away": 2}
        r = fdo.parse_matches(_payload(m), "UCL")[0]
        self.assertEqual((r["home_goals"], r["away_goals"]), (1, 1))

    def test_empty_payload_is_not_an_error(self):
        self.assertEqual(fdo.parse_matches({"matches": []}, "E0"), [])
        self.assertEqual(fdo.parse_matches({}, "E0"), [])


class TokenTests(unittest.TestCase):
    def test_unavailable_without_a_token(self):
        with patch.dict("os.environ", {"FOOTBALL_DATA_ORG_KEY": ""}, clear=False):
            self.assertFalse(fdo.available())

    def test_get_raises_a_helpful_error_without_a_token(self):
        with patch.dict("os.environ", {"FOOTBALL_DATA_ORG_KEY": ""}, clear=False):
            with self.assertRaises(fdo.MissingToken) as ctx:
                fdo._get("/competitions/PL/matches", {})
        self.assertIn("FOOTBALL_DATA_ORG_KEY", str(ctx.exception))


class CacheTests(unittest.TestCase):
    """The bug this prevents: api_football.py caches with no expiry at all, so
    the first fetch of a live season freezes it permanently."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(fdo, "RAW_FDORG", __import__("pathlib").Path(self.tmp.name))
        p.start()
        self.addCleanup(p.stop)

    def test_a_fresh_cache_file_is_reused_without_a_request(self):
        dest = fdo._cache_path("E0")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_payload(_match("A FC", "B FC", 1, 0))))
        with patch.object(fdo, "_get") as get:
            out = fdo.fetch_matches("E0", ttl=3600)
        get.assert_not_called()
        self.assertEqual(out["count"], 1)

    def test_a_stale_cache_file_triggers_a_refetch(self):
        dest = fdo._cache_path("E0")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_payload()))
        old = time.time() - 7200
        import os
        os.utime(dest, (old, old))
        fresh = _payload(_match("A FC", "B FC", 2, 2))
        with patch.object(fdo, "_get", return_value=fresh) as get:
            out = fdo.fetch_matches("E0", ttl=900)
        get.assert_called_once()
        self.assertEqual(out["count"], 1)
        # and the refreshed payload replaced the stale file
        self.assertEqual(json.loads(dest.read_text())["count"], 1)

    def test_ttl_zero_always_refetches(self):
        dest = fdo._cache_path("E0")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_payload()))
        with patch.object(fdo, "_get", return_value=_payload()) as get:
            fdo.fetch_matches("E0", ttl=0)
        get.assert_called_once()

    def test_a_failed_refetch_falls_back_to_the_stale_cache(self):
        # Better a table that is a few hours old than no table at all.
        dest = fdo._cache_path("E0")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_payload(_match("A FC", "B FC", 1, 0))))
        old = time.time() - 7200
        import os
        os.utime(dest, (old, old))
        with patch.object(fdo, "_get", side_effect=RuntimeError("503")):
            out = fdo.fetch_matches("E0", ttl=900)
        self.assertEqual(out["count"], 1)

    def test_a_failed_fetch_with_no_cache_raises(self):
        with patch.object(fdo, "_get", side_effect=RuntimeError("503")):
            with self.assertRaises(RuntimeError):
                fdo.fetch_matches("E0", ttl=900)

    def test_unknown_competition_is_rejected(self):
        with self.assertRaises(ValueError):
            fdo.fetch_matches("NOPE")


class CompetitionMapTests(unittest.TestCase):
    def test_the_three_displayed_competitions_are_mapped(self):
        # Verified against the live /v4/competitions list on 2026-09-13:
        # these are the free-tier (TIER_ONE) codes.
        self.assertEqual(fdo.COMPETITIONS["E0"], "PL")
        self.assertEqual(fdo.COMPETITIONS["SP1"], "PD")
        self.assertEqual(fdo.COMPETITIONS["UCL"], "CL")

    def test_every_mapped_competition_matches_a_configured_league(self):
        from futball import config
        for code in fdo.COMPETITIONS:
            self.assertIn(code, config.LEAGUES)


if __name__ == "__main__":
    unittest.main()
