from scripts.stats_request import game_log


def test_rate_limit_retry_after_is_bounded():
    pauses = []
    class Throttled(Exception):
        response = type('Response', (), {'headers': {'Retry-After': '120'}})()
    calls = 0
    def endpoint(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Throttled()
        return type('Log', (), {'get_data_frames': lambda self: [[{'ok': True}]]})()
    assert game_log(endpoint, pause=pauses.append) == [{'ok': True}]
    assert pauses == [30]
