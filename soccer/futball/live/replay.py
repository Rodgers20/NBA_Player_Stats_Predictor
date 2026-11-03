"""Replay provider — real matches on an accelerated clock.

This is the answer to a real constraint: live in-play data requires an API key
the project does not have yet. Rather than fake it with generated numbers, this
provider takes a genuine ingested match and re-emits its events in match-clock
order, scaled to wall-clock time.

Everything a viewer sees is true: the goals happened, the xG was modelled by
StatsBomb, the pass at 61.0/40.1 was played by that player at that second. Only
the pace is synthetic. That makes the whole realtime path — WebSocket fan-out,
reconnect, incremental chart updates, 3D scene streaming — testable end to end
today, and swappable for a genuine feed the moment a key exists.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from .. import marts
from .provider import LiveMessage

log = logging.getLogger(__name__)

# Event types worth pushing to a live feed. The full stream is ~3,600 events per
# match; a live ticker wants the meaningful ones plus a periodic stats refresh.
NOTABLE = {
    "Shot", "Substitution", "Foul Committed", "Bad Behaviour",
    "Own Goal Against", "Offside", "Card",
}


class ReplayProvider:
    name = "replay"

    def __init__(self, speed: float = 60.0) -> None:
        """`speed` = how many match-seconds elapse per real second."""
        self.speed = max(1.0, speed)

    async def stream(self, match_id: int) -> AsyncIterator[LiveMessage]:
        match = marts.match(match_id)
        if not match:
            raise ValueError(f"match {match_id} not found")

        events = marts.match_events(match_id)
        momentum = marts.momentum(match_id)
        momentum_by_minute = {m["minute"]: m for m in momentum}

        seq = 0

        def msg(kind: str, payload: dict[str, Any]) -> LiveMessage:
            nonlocal seq
            seq += 1
            return LiveMessage(seq=seq, kind=kind, match_id=match_id, payload=payload)

        yield msg("state", {
            "state": "kickoff",
            "home_team": match["home_team"], "away_team": match["away_team"],
            "stadium": match["stadium"], "referee": match["referee"],
            "competition": match["competition"], "stage": match["competition_stage"],
            "match_date": match["match_date"],
            "provider": self.name, "speed": self.speed,
        })

        home_score = away_score = 0
        pending = [e for e in events if e["period"] in (1, 2, 3, 4)]
        cursor = 0
        last_minute = max((e["minute"] for e in pending), default=90)
        total_minutes = max(last_minute + 2, 92)

        for minute in range(total_minutes + 1):
            # Emit every event that occurs in this match minute.
            while cursor < len(pending) and pending[cursor]["minute"] <= minute:
                e = pending[cursor]; cursor += 1
                if e["kind"] in ("goal", "own_goal"):
                    if e["team"] == match["home_team"]:
                        if e["kind"] == "goal":
                            home_score += 1
                        else:
                            away_score += 1
                    else:
                        if e["kind"] == "goal":
                            away_score += 1
                        else:
                            home_score += 1
                yield msg("event", {
                    **e,
                    "home_score": home_score,
                    "away_score": away_score,
                })

            yield msg("clock", {
                "minute": minute,
                "period": 1 if minute <= 45 else 2,
                "home_score": home_score,
                "away_score": away_score,
                "status": "live",
            })

            m = momentum_by_minute.get(minute)
            if m:
                yield msg("momentum", m)

            await asyncio.sleep(60.0 / self.speed)

        yield msg("state", {
            "state": "full_time",
            "home_score": home_score,
            "away_score": away_score,
            "official_home": match["home_score"],
            "official_away": match["away_score"],
        })
