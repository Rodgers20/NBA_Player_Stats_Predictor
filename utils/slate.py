"""Which date's games the app should show: today's, else the next day that has any."""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
LOOKAHEAD_DAYS = 3


def today_et():
    return datetime.now(ET).date().isoformat()


def next_slate(games_on, start=None, days=LOOKAHEAD_DAYS):
    """(date, games) for the first day from `start` with games; (start, []) if none within `days`."""
    start = start or today_et()
    for offset in range(days + 1):
        day = (date.fromisoformat(start) + timedelta(days=offset)).isoformat()
        games = games_on(day)
        if games:
            return day, games
    return start, []


def slate_date(league):
    """Date the props board should evaluate for this league."""
    if league == 'wnba':
        from utils.wnba_data_fetch import get_todays_wnba_games
        return next_slate(get_todays_wnba_games)[0]
    from utils.data_fetch import get_upcoming_games
    try:
        frame, target = get_upcoming_games()   # already falls back from today to tomorrow
        return str(target) if target and frame is not None and not frame.empty else today_et()
    except Exception:
        return today_et()
