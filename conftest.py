import sys
import os

import pytest

os.environ["NBA_DISABLE_BACKGROUND"] = "1"
os.environ["ODDS_AUTO_REFRESH"] = "0"

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """Clear cross-test module-level caches.

    utils.wnba_data_fetch caches the WNBA schedule by date at module scope.
    Without this, a test that fetches a date leaks its stubbed result into any
    later test using the same date, which silently ignores that test's own
    fixture data.
    """
    try:
        from utils.wnba_data_fetch import clear_schedule_cache
    except Exception:
        yield
        return

    clear_schedule_cache()
    yield
    clear_schedule_cache()


@pytest.fixture(autouse=True)
def _slate_is_today(monkeypatch):
    """Keep route tests off the schedule network; fallback logic is tested through next_slate."""
    from utils import slate
    monkeypatch.setattr(slate, "slate_date", lambda league: slate.today_et())


@pytest.fixture(autouse=True)
def _isolate_personal_storage(tmp_path, monkeypatch):
    from utils import odds_budget
    monkeypatch.setattr(odds_budget, "DB_PATH", tmp_path / "personal.sqlite3")
    from utils import wnba_injuries
    monkeypatch.setattr(wnba_injuries, "_CACHE_FILE", tmp_path / "wnba_injuries.json")
