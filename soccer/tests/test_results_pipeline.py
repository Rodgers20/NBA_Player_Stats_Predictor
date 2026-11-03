"""End-to-end: a football-data.org payload through to a deduped results frame.

Every other merge test hands both sides pre-agreed `date` / `home_team` strings,
so none of them can catch the two pipelines *deriving* those fields differently.
This one starts from a raw API payload on one side and a canonical lake row on
the other, and asserts they actually meet.
"""
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from futball.sources import football_data_org as fdo
from futball.transform import results


def _api_match(home, away, hg, ag, utc, *, season_start="2026-08-15", matchday=4):
    return {
        "utcDate": utc, "status": "FINISHED", "matchday": matchday,
        "stage": "REGULAR_SEASON",
        "season": {"startDate": season_start, "endDate": "2027-05-23"},
        "homeTeam": {"name": home}, "awayTeam": {"name": away},
        "score": {"winner": "HOME", "duration": "REGULAR",
                  "fullTime": {"home": hg, "away": ag},
                  "halfTime": {"home": 0, "away": 0}},
    }


def _canonical(code, season, date, home, away, hg, ag, **extra):
    return results._finish(results._row(
        source="football-data.co.uk" if code != "UCL" else "openfootball",
        competition_code=code, competition="X", country="Y", season=season,
        date=date, home_team=home, away_team=away,
        home_goals=hg, away_goals=ag, **extra))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(fdo, "RAW_FDORG", pathlib.Path(self.tmp.name))
        p.start()
        self.addCleanup(p.stop)

    def _write(self, code, *matches):
        dest = fdo._cache_path(code)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"count": len(matches), "matches": list(matches)}))

    def test_a_new_matchday_merges_without_duplicating_the_old_one(self):
        # The lake already has matchday 3, from football-data.co.uk, with shots.
        # Every club involved must already be known — in the real lake there are
        # 40 canonical Premier League clubs across 16 seasons, so this is what
        # the resolver actually sees.
        canonical = [
            _canonical("E0", "2026/27", "2026-09-06", "Arsenal", "Chelsea", 2, 1,
                       home_shots=16, away_shots=13, odds_home=1.69, time="16:30"),
            _canonical("E0", "2026/27", "2026-09-05", "Nott'm Forest", "Tottenham",
                       0, 0, home_shots=8, away_shots=11),
        ]
        # football-data.org reports matchday 3 again (already known) plus
        # matchday 4 (new), under its own spellings.
        self._write(
            "E0",
            _api_match("Arsenal FC", "Chelsea FC", 2, 1, "2026-09-06T16:30:00Z", matchday=3),
            _api_match("Nottingham Forest FC", "Tottenham Hotspur FC", 1, 1,
                       "2026-09-12T14:00:00Z"),
        )

        supplementary, dropped, rows_dropped = results.parse_football_data_org(canonical)
        self.assertEqual(dropped, {})
        self.assertEqual(rows_dropped, 0)

        merged = results.dedupe(
            pl.DataFrame(canonical + supplementary, schema=results.ROW_SCHEMA))

        self.assertEqual(merged.height, 3, "the shared match must not duplicate")
        # The already-known match kept its football-data.co.uk detail.
        arsenal = merged.filter(pl.col("home_team") == "Arsenal").to_dicts()[0]
        self.assertEqual(arsenal["home_shots"], 16)
        self.assertEqual(arsenal["source"], "football-data.co.uk")
        # The new match arrived, canonically spelled.
        forest = merged.filter(pl.col("date") == "2026-09-12").to_dicts()[0]
        self.assertEqual(forest["home_team"], "Nott'm Forest")
        self.assertEqual(forest["away_team"], "Tottenham")
        self.assertEqual(forest["source"], "football-data.org")

    def test_independently_derived_keys_actually_match(self):
        """The two pipelines must agree on date, season and name, or dedupe
        never sees a collision in the first place."""
        canonical = [_canonical("E0", "2026/27", "2026-09-12", "Man United",
                                "Man City", 1, 1, home_shots=10)]
        self._write("E0", _api_match("Manchester United FC", "Manchester City FC",
                                     1, 1, "2026-09-12T16:30:00Z"))
        supplementary, _, _ = results.parse_football_data_org(canonical)
        self.assertEqual(len(supplementary), 1)
        self.assertEqual(supplementary[0]["match_key"], canonical[0]["match_key"])

    def test_a_season_no_other_source_covers_runs_lenient(self):
        # Champions League 2026/27: nothing canonical exists, so an unfamiliar
        # club must survive rather than be dropped.
        canonical = [_canonical("UCL", "2025-26", "2025-09-16", "Arsenal",
                                "Chelsea", 2, 0)]
        self._write("UCL", _api_match("Qarabag FK", "Arsenal FC", 0, 2,
                                      "2026-09-16T20:00:00Z"))
        supplementary, dropped, rows_dropped = results.parse_football_data_org(canonical)
        self.assertEqual(len(supplementary), 1)
        self.assertEqual(dropped, {})
        self.assertEqual(rows_dropped, 0)
        self.assertEqual(supplementary[0]["home_team"], "Qarabag FK")
        self.assertEqual(supplementary[0]["season"], "2026-27")

    def test_a_season_another_source_covers_flips_to_strict(self):
        """The moment openfootball publishes CL 2026-27, the guard turns itself
        back on for that season — no code change, no config flag."""
        canonical = [
            _canonical("UCL", "2026-27", "2026-09-16", "Arsenal", "Chelsea", 2, 0),
        ]
        self._write("UCL", _api_match("Qarabag FK", "Arsenal FC", 0, 2,
                                      "2026-09-17T20:00:00Z"))
        supplementary, dropped, rows_dropped = results.parse_football_data_org(canonical)
        self.assertEqual(supplementary, [], "unresolved club must now be dropped")
        self.assertEqual(dropped, {"UCL": ["Qarabag FK"]})
        self.assertEqual(rows_dropped, 1)

    def test_strictness_is_decided_per_season_not_per_competition(self):
        # One covered season and one uncovered season in the same competition:
        # the covered one is strict, the uncovered one is not.
        canonical = [
            _canonical("UCL", "2025-26", "2025-09-16", "Arsenal", "Chelsea", 2, 0),
        ]
        self._write(
            "UCL",
            # 2025-26 is covered -> unknown club dropped
            _api_match("Unknown Old Club", "Arsenal FC", 0, 1,
                       "2025-10-01T20:00:00Z", season_start="2025-08-15"),
            # 2026-27 is not covered -> unknown club kept
            _api_match("Unknown New Club", "Arsenal FC", 1, 1,
                       "2026-10-01T20:00:00Z", season_start="2026-08-15"),
        )
        supplementary, dropped, rows_dropped = results.parse_football_data_org(canonical)
        seasons = sorted(r["season"] for r in supplementary)
        self.assertEqual(seasons, ["2026-27"])
        self.assertEqual(dropped, {"UCL": ["Unknown Old Club"]})
        self.assertEqual(rows_dropped, 1)

    def test_a_club_with_no_canonical_spelling_is_dropped_not_guessed(self):
        """A deliberate trade, inherited from teams.py's stated policy.

        A club the lake has never seen — a promotion whose first match lands in
        the window before football-data.co.uk publishes — is dropped and logged
        rather than admitted under a foreign spelling. Admitting it risks a
        second copy of a club that IS already known under a name the resolver
        merely failed to match, and a duplicated club silently corrupts the
        table. The drop is visible in the refresh log and self-heals on the next
        football-data.co.uk publish.
        """
        canonical = [_canonical("E0", "2026/27", "2026-09-06", "Arsenal",
                                "Chelsea", 2, 1, home_shots=16)]
        self._write("E0", _api_match("Newly Promoted Rovers FC", "Arsenal FC",
                                     0, 3, "2026-09-12T14:00:00Z"))
        supplementary, dropped, rows_dropped = results.parse_football_data_org(canonical)
        self.assertEqual(supplementary, [])
        self.assertEqual(dropped, {"E0": ["Newly Promoted Rovers FC"]})
        self.assertEqual(rows_dropped, 1)

    def test_no_cached_payload_is_a_no_op(self):
        supplementary, dropped, rows_dropped = results.parse_football_data_org([])
        self.assertEqual((supplementary, dropped, rows_dropped), ([], {}, 0))


if __name__ == "__main__":
    unittest.main()
