import asyncio
import unittest
from unittest.mock import patch
from futball.api.app import ws_match
from futball.live.hub import LiveHub, MatchStream, QUEUE_SIZE
from futball.live.provider import LiveMessage

class Socket:
    def __init__(self, disconnect=True, until=0):
        self.disconnected=asyncio.Event()
        if disconnect: self.disconnected.set()
        self.messages=[];self.until=until;self.code=None
    async def accept(self): pass
    async def receive(self):
        await self.disconnected.wait()
        return {'type':'websocket.disconnect'}
    async def send_json(self,message):
        self.messages.append(message)
        if self.until and len(self.messages)>=self.until:self.disconnected.set()
    async def close(self,code=1000,reason=''):
        self.code=code;self.disconnected.set()

class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_disconnect_releases_subscription(self):
        hub=LiveHub()
        with patch('futball.api.app.HUB',hub),patch('futball.api.app.marts.match',return_value={'match_id':1}):
            await asyncio.wait_for(ws_match(Socket(),1),1)
        self.assertEqual(hub.status()[0]['subscribers'],0)

    async def test_reconnect_receives_only_missing_messages(self):
        hub=LiveHub();hub.subscribe(1)
        stream=hub._streams[1]
        for _ in range(3):hub._dispatch(stream,LiveMessage(1,'clock',1,{}))
        socket=Socket(False,until=2)
        with patch('futball.api.app.HUB',hub),patch('futball.api.app.marts.match',return_value={'match_id':1}):
            await asyncio.wait_for(ws_match(socket,1,since=1),1)
        self.assertEqual([m['seq'] for m in socket.messages],[2,3])
        self.assertEqual(hub.status()[0]['subscribers'],1) # only the original test subscription

    async def test_unknown_match_is_rejected_without_subscription(self):
        hub=LiveHub();socket=Socket()
        with patch('futball.api.app.HUB',hub),patch('futball.api.app.marts.match',return_value=None):
            await ws_match(socket,999)
        self.assertEqual(socket.code,1008);self.assertEqual(hub.status(),[])

    async def test_expired_history_rejected_instead_of_silent_gap(self):
        hub=LiveHub();hub.subscribe(1);stream=hub._streams[1]
        for _ in range(4):hub._dispatch(stream,LiveMessage(1,'clock',1,{}))
        stream.backlog.popleft();stream.backlog.popleft()
        self.assertFalse(hub.can_resume(1,1));self.assertTrue(hub.can_resume(1,2))
        socket=Socket()
        with patch('futball.api.app.HUB',hub),patch('futball.api.app.marts.match',return_value={'match_id':1}):
            await ws_match(socket,1,since=1)
        self.assertEqual(socket.code,1008)

    async def test_slow_subscriber_gets_disconnect_signal(self):
        hub=LiveHub();q=hub.subscribe(1)
        for _ in range(QUEUE_SIZE):q.put_nowait({})
        hub._dispatch(hub._streams[1],LiveMessage(1,'clock',1,{}))
        self.assertIsNone(q.get_nowait());self.assertEqual(hub.status()[0]['subscribers'],0)

    async def test_stop_then_immediate_restart_keeps_monotonic_sequence(self):
        class Provider:
            def __init__(self,**kwargs):pass
            async def stream(self,match_id):
                yield LiveMessage(1,'state',match_id,{'state':'kickoff'})
                await asyncio.Event().wait()
        hub=LiveHub()
        with patch('futball.live.hub.ReplayProvider',Provider):
            hub.start(1);await asyncio.sleep(0)
            first=hub._streams[1].last_seq
            old=hub._streams[1].task
            hub.stop(1);hub.start(1);await asyncio.sleep(0)
            await asyncio.gather(old,return_exceptions=True)
            stream=hub._streams[1]
            self.assertEqual(stream.state,'running')
            self.assertGreater(stream.last_seq,first)
            current=stream.task;hub.stop(1)
            await asyncio.gather(current,return_exceptions=True)
