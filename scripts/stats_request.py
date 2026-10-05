"""Small, bounded NBA Stats request helper for scheduled data refreshes."""
import time


def game_log(endpoint, *, attempts=3, pause=time.sleep, **kwargs):
    for attempt in range(attempts):
        try:
            return endpoint(**kwargs).get_data_frames()[0]
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            response = getattr(exc, 'response', None)
            headers = getattr(response, 'headers', {}) or {}
            retry_after = headers.get('Retry-After')
            try:
                delay = min(30, max(1, float(retry_after))) if retry_after else 2 ** (attempt + 1)
            except (TypeError, ValueError):
                delay = 2 ** (attempt + 1)
            pause(delay)
