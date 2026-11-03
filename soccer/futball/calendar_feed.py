"""iCalendar feeds for fixtures.

Every kick-off is already in the silver fixture layer with a UTC timestamp, so
this is pure formatting — no source, no key, no request budget.

Two details make the difference between a feed people subscribe to and one they
delete:

* **A stable UID per fixture.** Calendar clients key on it. Regenerate a feed
  with fresh UIDs and every match appears twice.
* **A SEQUENCE that rises when the kick-off moves.** Without it a client that
  has already imported an event ignores the corrected time.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from . import config, league

PRODID = "-//FUTBALL//Football Intelligence//EN"

# A match blocks this much of a calendar: 90 minutes plus half-time and the
# stoppage that realistically follows.
MATCH_BLOCK = timedelta(minutes=config.MATCH_DURATION_MINUTES)


def _escape(text: str) -> str:
    """iCalendar escaping: backslash, semicolon, comma and newline."""
    return (
        (text or "")
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    """RFC 5545 caps a content line at 75 octets; continuations start with a space."""
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    out, chunk = [], b""
    for ch in line:
        encoded = ch.encode("utf-8")
        if len(chunk) + len(encoded) > (75 if not out else 74):
            out.append(chunk.decode("utf-8"))
            chunk = b""
        chunk += encoded
    if chunk:
        out.append(chunk.decode("utf-8"))
    return "\r\n ".join(out)


def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _uid(fixture_key: str) -> str:
    """Stable per fixture, so a re-subscribe updates rather than duplicates."""
    digest = hashlib.sha256(fixture_key.encode("utf-8")).hexdigest()[:24]
    return f"{digest}@futball"


def _sequence(kickoff_iso: str | None) -> int:
    """Rises when the kick-off changes, so clients accept the correction.

    Derived from the kick-off itself rather than stored: any change to the time
    changes the number, which is exactly the condition a bump should signal.
    """
    if not kickoff_iso:
        return 0
    return int(hashlib.sha256(kickoff_iso.encode("utf-8")).hexdigest()[:6], 16) % 1000


def _events(fixtures: list[dict], now: datetime) -> list[str]:
    lines: list[str] = []
    for f in fixtures:
        kickoff = f.get("kickoff_utc")
        if not kickoff:
            continue
        try:
            start = datetime.fromisoformat(kickoff)
        except ValueError:
            continue
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)

        home, away = f.get("home_team"), f.get("away_team")
        if not home or not away:
            continue
        meta = config.LEAGUES.get(f.get("competition_code", ""), {})

        description = [meta.get("name", f.get("competition_code", ""))]
        if f.get("round"):
            description.append(str(f["round"]))
        if f.get("referee"):
            description.append(f"Referee: {f['referee']}")

        lines += [
            "BEGIN:VEVENT",
            f"UID:{_uid(f.get('fixture_key') or f'{home}|{away}|{kickoff}')}",
            f"SEQUENCE:{_sequence(kickoff)}",
            f"DTSTAMP:{_stamp(now)}",
            f"DTSTART:{_stamp(start)}",
            f"DTEND:{_stamp(start + MATCH_BLOCK)}",
            _fold(f"SUMMARY:{_escape(f'{home} v {away}')}"),
            _fold(f"DESCRIPTION:{_escape(' · '.join(description))}"),
            "STATUS:CONFIRMED",
            "TRANSP:TRANSPARENT",
            "END:VEVENT",
        ]
    return lines


def competition_calendar(code: str, days: int = 120,
                         now: datetime | None = None) -> str:
    """An `.ics` feed for one competition's upcoming fixtures."""
    now = now or datetime.now(timezone.utc)
    meta = config.LEAGUES.get(code, {})
    name = meta.get("name", code)
    fixtures = league.fixtures(code=code, days=days, limit=1000)
    return _document(f"FUTBALL — {name}", _events(fixtures, now))


def team_calendar(team: str, days: int = 120,
                  now: datetime | None = None) -> str:
    """An `.ics` feed for one club, across every competition it appears in."""
    now = now or datetime.now(timezone.utc)
    fixtures = [
        f for f in league.fixtures(days=days, limit=2000)
        if team in (f.get("home_team"), f.get("away_team"))
    ]
    return _document(f"FUTBALL — {team}", _events(fixtures, now))


def _document(name: str, events: list[str]) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{PRODID}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        _fold(f"X-WR-CALNAME:{_escape(name)}"),
        # Poll interval for clients that subscribe rather than import.
        "X-PUBLISHED-TTL:PT6H",
        "REFRESH-INTERVAL;VALUE=DURATION:PT6H",
        *events,
        "END:VCALENDAR",
    ]
    # RFC 5545 requires CRLF line endings.
    return "\r\n".join(lines) + "\r\n"
