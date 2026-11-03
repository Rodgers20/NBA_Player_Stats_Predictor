"""Rate limiting, entitlement errors and date-boundary safety.

These cover the parts of the adapter that only misbehave against the real API:
the free tier is 10 requests/minute, and exceeding it earns a 429 rather than a
queue. None of it is exercised by the parsing tests.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from futball.sources import football_data_org as fdo


class ThrottleTests(unittest.TestCase):
    def setUp(self):
        # Each test starts from an un-throttled state.
        fdo._last_request = 0.0
        self.addCleanup(setattr, fdo, "_last_request", 0.0)

    def test_consecutive_calls_are_spaced_by_the_minimum_interval(self):
        with patch.object(fdo, "MIN_REQUEST_INTERVAL", 0.05):
            fdo._throttle()
            start = time.monotonic()
            fdo._throttle()
            elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.05)

    def test_the_first_call_after_a_long_idle_does_not_sleep(self):
        fdo._last_request = time.monotonic() - 1000
        with patch.object(fdo, "MIN_REQUEST_INTERVAL", 5.0):
            start = time.monotonic()
            fdo._throttle()
            self.assertLess(time.monotonic() - start, 0.05)

    def test_concurrent_callers_are_serialised_not_raced(self):
        """The lock is held across the sleep, so N threads take (N-1) intervals.

        Without that, every thread reads the same `_last_request`, all sleep the
        same amount, and all fire at once — which is exactly how a 10/min limit
        gets blown by a burst.
        """
        interval = 0.05
        threads = 4
        with patch.object(fdo, "MIN_REQUEST_INTERVAL", interval):
            fdo._throttle()  # prime it
            start = time.monotonic()
            workers = [threading.Thread(target=fdo._throttle) for _ in range(threads)]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
            elapsed = time.monotonic() - start
        self.assertGreaterEqual(
            elapsed, interval * threads * 0.8,
            "throttle did not serialise concurrent callers")


class RateLimitResponseTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict("os.environ", {"FOOTBALL_DATA_ORG_KEY": "test-token"})
        patcher.start()
        self.addCleanup(patcher.stop)
        throttle = patch.object(fdo, "_throttle")
        throttle.start()
        self.addCleanup(throttle.stop)

    def _client_returning(self, *responses):
        client = MagicMock()
        client.get.side_effect = list(responses)
        ctx = MagicMock()
        ctx.__enter__.return_value = client
        return ctx, client

    def _response(self, status, *, json_body=None, headers=None):
        r = MagicMock()
        r.status_code = status
        r.headers = headers or {}
        r.json.return_value = json_body or {}
        r.raise_for_status.return_value = None
        return r

    def test_429_waits_for_the_reset_header_then_retries_once(self):
        limited = self._response(429, headers={"X-RequestCounter-Reset": "3"})
        ok = self._response(200, json_body={"count": 1, "matches": []})
        ctx, client = self._client_returning(limited, ok)
        with patch.object(fdo.httpx, "Client", return_value=ctx), \
             patch.object(fdo.time, "sleep") as slept:
            body = fdo._get("/competitions/PL/matches", {})
        self.assertEqual(body["count"], 1)
        self.assertEqual(client.get.call_count, 2, "should retry exactly once")
        slept.assert_called_once()
        self.assertAlmostEqual(slept.call_args.args[0], 4.0)  # reset + 1s margin

    def test_a_missing_reset_header_falls_back_to_a_bounded_wait(self):
        limited = self._response(429, headers={})
        ok = self._response(200, json_body={})
        ctx, _ = self._client_returning(limited, ok)
        with patch.object(fdo.httpx, "Client", return_value=ctx), \
             patch.object(fdo.time, "sleep") as slept:
            fdo._get("/competitions/PL/matches", {})
        self.assertLessEqual(slept.call_args.args[0], 61)

    def test_403_raises_not_entitled_rather_than_a_bare_runtime_error(self):
        ctx, _ = self._client_returning(self._response(403))
        with patch.object(fdo.httpx, "Client", return_value=ctx):
            with self.assertRaises(fdo.NotEntitled):
                fdo._get("/competitions/ELC/matches", {})


class ConfigErrorsBypassTheStaleCacheTests(unittest.TestCase):
    """A stale payload must never paper over a broken token or wrong plan.

    Otherwise the dashboard keeps serving data that quietly stopped updating,
    which is worse than an error, because nothing looks wrong.
    """

    def setUp(self):
        import pathlib
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(fdo, "RAW_FDORG", pathlib.Path(self.tmp.name))
        p.start()
        self.addCleanup(p.stop)

        import json
        dest = fdo._cache_path("E0")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"count": 0, "matches": []}))
        old = time.time() - 7200
        import os
        os.utime(dest, (old, old))

    def test_missing_token_propagates_even_with_a_cache_present(self):
        with patch.object(fdo, "_get", side_effect=fdo.MissingToken("no token")):
            with self.assertRaises(fdo.MissingToken):
                fdo.fetch_matches("E0", ttl=900)

    def test_not_entitled_propagates_even_with_a_cache_present(self):
        with patch.object(fdo, "_get", side_effect=fdo.NotEntitled("403")):
            with self.assertRaises(fdo.NotEntitled):
                fdo.fetch_matches("E0", ttl=900)

    def test_a_transport_error_still_falls_back(self):
        with patch.object(fdo, "_get", side_effect=OSError("connection reset")):
            self.assertEqual(fdo.fetch_matches("E0", ttl=900)["count"], 0)


class DateBoundaryTests(unittest.TestCase):
    """`match_key` embeds the date, so UTC-vs-local skew would split a fixture."""

    def _row(self, utc):
        payload = {"matches": [{
            "utcDate": utc, "status": "FINISHED", "matchday": 5,
            "season": {"startDate": "2026-08-15"},
            "homeTeam": {"name": "A FC"}, "awayTeam": {"name": "B FC"},
            "score": {"fullTime": {"home": 1, "away": 0},
                      "halfTime": {"home": 0, "away": 0}},
        }]}
        return fdo.parse_matches(payload, "SP1")[0]

    def test_utc_date_matches_local_date_for_late_kickoffs(self):
        # The latest realistic European kick-off: 22:00 CEST == 20:00 UTC.
        # Same calendar day, so football-data.co.uk's local date agrees.
        self.assertEqual(self._row("2026-09-12T20:00:00Z")["date"], "2026-09-12")

    def test_the_assumption_would_break_past_midnight_utc(self):
        # Documents the boundary rather than asserting today's behaviour is
        # wrong: a 01:00 UTC kick-off IS 03:00 CEST the same night, and would be
        # filed by football-data.co.uk under the previous day. No European
        # fixture kicks off then, which is why UTC truncation is safe here — but
        # a competition outside UTC..UTC+2 would need a timezone conversion.
        self.assertEqual(self._row("2026-09-13T01:00:00Z")["date"], "2026-09-13")


if __name__ == "__main__":
    unittest.main()
