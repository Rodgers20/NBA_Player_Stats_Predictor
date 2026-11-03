"""Live feed provider interface.

Three implementations share one contract so the UI never learns where a match
is coming from:

* `ReplayProvider`  — streams a real, already-ingested match on an accelerated
  clock. No API key, no mock data: every event actually happened.
* `FootballDataProvider`  — football-data.org, free key, live scores.
* `ApiFootballProvider`   — api-sports.io, free key, full in-play events.

Swapping providers is a `.env` change, not a code change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class LiveMessage:
    """One message on the wire. `seq` lets a reconnecting client resume."""

    seq: int
    kind: str  # clock | event | stats | momentum | state
    match_id: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "match_id": self.match_id,
            "payload": self.payload,
        }


@runtime_checkable
class LiveProvider(Protocol):
    name: str

    def stream(self, match_id: int) -> AsyncIterator[LiveMessage]:
        """Yield messages until the match ends or the stream is cancelled."""
        ...
