import sys
import os

import pytest

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
