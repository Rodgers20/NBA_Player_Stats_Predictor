import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from futball import cli


def _args(**kw):
    base = {"seasons": 2, "retrain": False, "force": False, "squad_limit": 12}
    base.update(kw)
    return SimpleNamespace(**base)


# Everything cmd_refresh touches that would otherwise hit the network or disk.
_STUBS = (
    ('futball.config.ensure_dirs', {}),
    ('futball.sources.football_data.ingest',
     {"return_value": SimpleNamespace(unavailable=[])}),
    ('futball.sources.openfootball.ingest_champions_league', {}),
    ('futball.transform.results.build', {"return_value": {}}),
    ('futball.http.PoliteClient', {}),
    ('futball.sources.fixtures.fetch_fixtures_csv', {}),
    ('futball.sources.fixtures.fetch_season_schedules', {}),
    ('futball.transform.fixture_table.build', {}),
    # Squad caching for player portraits; a no-op while the 7-day TTL is fresh,
    # but stubbed so the test never reaches API-Football.
    ('futball.headshots.backfill_squads', {"return_value": {}}),
)


class RefreshTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.mocks = {
            target.rsplit('.', 1)[-1] if 'fixture_table' not in target else 'fixture_build':
                stack.enter_context(patch(target, **kw))
            for target, kw in _STUBS
        }
        self.stack = stack

    def _patch(self, target, **kw):
        return self.stack.enter_context(patch(target, **kw))

    def test_refresh_updates_fixtures_without_retraining(self):
        self._patch('futball.sources.football_data_org.available', return_value=False)
        self.assertEqual(cli.cmd_refresh(_args()), 0)
        self.mocks['fetch_fixtures_csv'].assert_called_once()
        self.mocks['fetch_season_schedules'].assert_called_once()
        self.mocks['fixture_build'].assert_called_once()

    def test_football_data_org_is_fetched_when_a_token_is_present(self):
        self._patch('futball.sources.football_data_org.available', return_value=True)
        ingest = self._patch('futball.sources.football_data_org.ingest',
                             return_value={"fetched": 3})
        self.assertEqual(cli.cmd_refresh(_args()), 0)
        ingest.assert_called_once()

    def test_football_data_org_is_skipped_without_a_token(self):
        # The whole point of the adapter being optional: no key, no crash.
        self._patch('futball.sources.football_data_org.available', return_value=False)
        ingest = self._patch('futball.sources.football_data_org.ingest')
        self.assertEqual(cli.cmd_refresh(_args()), 0)
        ingest.assert_not_called()

    def test_force_bypasses_the_cache_ttl(self):
        self._patch('futball.sources.football_data_org.available', return_value=True)
        ingest = self._patch('futball.sources.football_data_org.ingest', return_value={})
        cli.cmd_refresh(_args(force=True))
        self.assertEqual(ingest.call_args.kwargs["ttl"], 0)

    def test_default_run_uses_the_normal_ttl(self):
        self._patch('futball.sources.football_data_org.available', return_value=True)
        ingest = self._patch('futball.sources.football_data_org.ingest', return_value={})
        cli.cmd_refresh(_args())
        self.assertGreater(ingest.call_args.kwargs["ttl"], 0)


if __name__ == "__main__":
    unittest.main()
