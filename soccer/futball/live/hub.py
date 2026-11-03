"""WebSocket fan-out hub.

One producer task per live match; many subscriber queues. A bounded backlog is
retained per match so a client that drops and reconnects with `?since=<seq>`
gets the messages it missed instead of a hole in its timeline.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field

from .provider import LiveMessage
from .replay import ReplayProvider

log = logging.getLogger(__name__)

BACKLOG_SIZE = 2000
QUEUE_SIZE = 500


@dataclass
class MatchStream:
    match_id: int
    speed: float
    task: asyncio.Task | None = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    backlog: deque[dict] = field(default_factory=lambda: deque(maxlen=BACKLOG_SIZE))
    last_seq: int = 0
    state: str = "idle"


class LiveHub:
    def __init__(self) -> None:
        self._streams: dict[int, MatchStream] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ control ---

    def start(self, match_id: int, speed: float = 60.0) -> None:
        """Begin producing for `match_id`. Must be called from the event loop.

        A stream object may already exist because a client subscribed before
        pressing play. In that case we reuse it rather than replacing it —
        replacing would orphan every queue already handed out to subscribers,
        and they would sit silently forever.
        """
        stream = self._streams.get(match_id)
        if stream is None:
            stream = MatchStream(match_id=match_id, speed=speed)
            self._streams[match_id] = stream

        if stream.task and not stream.task.done():
            log.info("match %s already streaming", match_id)
            return

        # Reuse the existing subscriber set; only reset the produced state.
        stream.speed = speed
        stream.backlog.clear()
        stream.task = asyncio.create_task(self._produce(stream))
        stream.state = "running"

    def stop(self, match_id: int) -> None:
        stream = self._streams.get(match_id)
        if stream and stream.task:
            stream.task.cancel()
            stream.task = None
            stream.state = "stopped"
            self._dispatch(stream, LiveMessage(0, "state", match_id, {"state": "stopped"}))

    def status(self) -> list[dict]:
        return [
            {
                "match_id": s.match_id,
                "state": s.state,
                "speed": s.speed,
                "subscribers": len(s.subscribers),
                "last_seq": s.last_seq,
            }
            for s in self._streams.values()
        ]

    # ---------------------------------------------------------- subscribe ---

    def subscribe(self, match_id: int) -> asyncio.Queue:
        stream = self._streams.setdefault(match_id, MatchStream(match_id, speed=60.0))
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        stream.subscribers.add(q)
        return q

    def unsubscribe(self, match_id: int, q: asyncio.Queue) -> None:
        stream = self._streams.get(match_id)
        if stream:
            stream.subscribers.discard(q)

    def can_resume(self, match_id: int, since: int) -> bool:
        if since == 0:
            return True
        stream = self._streams.get(match_id)
        if not stream or since < 0 or since > stream.last_seq:
            return False
        return not stream.backlog or since >= stream.backlog[0]["seq"] - 1

    def backlog(self, match_id: int, since: int = 0) -> list[dict]:
        stream = self._streams.get(match_id)
        if not stream:
            return []
        return [m for m in stream.backlog if m["seq"] > since]

    # ----------------------------------------------------------- producer ---

    async def _produce(self, stream: MatchStream) -> None:
        provider = ReplayProvider(speed=stream.speed)
        try:
            async for message in provider.stream(stream.match_id):
                self._dispatch(stream, message)
            if stream.task is asyncio.current_task():
                stream.state = "finished"
        except asyncio.CancelledError:
            if stream.task is asyncio.current_task():
                stream.state = "stopped"
            raise
        except Exception as exc:  # noqa: BLE001
            if stream.task is asyncio.current_task():
                stream.state = "error"
            log.exception("live producer failed for match %s: %s", stream.match_id, exc)

    def _dispatch(self, stream: MatchStream, message: LiveMessage) -> None:
        payload = message.to_json()
        stream.last_seq += 1
        payload["seq"] = stream.last_seq
        stream.backlog.append(payload)
        for q in list(stream.subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # A subscriber that cannot keep up is dropped rather than allowed
                # to stall the producer for everyone else. It can reconnect with
                # ?since=<seq> and recover from the backlog.
                log.warning("dropping slow subscriber on match %s", stream.match_id)
                stream.subscribers.discard(q)
                while not q.empty():
                    q.get_nowait()
                q.put_nowait(None)


HUB = LiveHub()
