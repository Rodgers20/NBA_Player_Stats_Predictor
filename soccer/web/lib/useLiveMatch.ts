"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, wsUrl } from "./api";
import type { MatchEvent, MomentumPoint } from "./types";

type Status = "idle" | "connecting" | "live" | "finished" | "error";

/**
 * Subscribes to the live match WebSocket.
 *
 * Reconnection resumes from the last sequence number seen, so a dropped
 * connection produces no hole in the event timeline — the server replays the
 * backlog past `since`. Without that a reconnect would silently lose goals.
 */
export function useLiveMatch(matchId: number, autoStart = false) {
  const [status, setStatus] = useState<Status>("idle");
  const [connected, setConnected] = useState(false);
  const [minute, setMinute] = useState(0);
  const [second, setSecond] = useState(0);
  const [homeScore, setHomeScore] = useState(0);
  const [awayScore, setAwayScore] = useState(0);
  const [events, setEvents] = useState<MatchEvent[]>([]);
  const [momentum, setMomentum] = useState<MomentumPoint[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const seqRef = useRef(0);
  const wantOpenRef = useRef(false);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const closeSocket = useCallback(() => {
    if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null; }
    socketRef.current?.close();
    socketRef.current = null;
    setConnected(false);
  }, []);

  const connect = useCallback(function reconnect() {
    if (typeof window === "undefined") return;
    if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(wsUrl(matchId, seqRef.current));
    socketRef.current = ws;

    ws.onopen = () => { if (socketRef.current === ws) setConnected(true); };

    ws.onmessage = (raw) => {
      if (socketRef.current !== ws) return;
      let msg: { seq: number; kind: string; payload: Record<string, unknown> };
      try {
        msg = JSON.parse(raw.data as string);
      } catch {
        return; // never let a malformed frame kill the stream
      }
      if (!Number.isInteger(msg.seq) || msg.seq <= seqRef.current) return;
      seqRef.current = msg.seq;

      switch (msg.kind) {
        case "state": {
          const state = msg.payload.state as string;
          if (state === "kickoff") { setStatus("live"); setEvents([]); setMomentum([]); setHomeScore(0); setAwayScore(0); setMinute(0); setSecond(0); }
          if (state === "stopped") { setStatus("idle"); wantOpenRef.current = false; ws.close(); }
          if (state === "full_time") { setStatus("finished"); wantOpenRef.current = false; }
          break;
        }
        case "clock":
          setMinute(msg.payload.minute as number);
          setSecond(0);
          setHomeScore(msg.payload.home_score as number);
          setAwayScore(msg.payload.away_score as number);
          setStatus("live");
          break;
        case "event":
          setEvents((prev) => [msg.payload as unknown as MatchEvent, ...prev].slice(0, 60));
          setHomeScore(msg.payload.home_score as number);
          setAwayScore(msg.payload.away_score as number);
          break;
        case "momentum":
          setMomentum((prev) => [...prev, msg.payload as unknown as MomentumPoint]);
          break;
      }
    };

    ws.onclose = (event) => {
      if (socketRef.current !== ws) return;
      if (event.code === 1008) { wantOpenRef.current = false; setStatus("error"); }
      setConnected(false);
      socketRef.current = null;
      if (wantOpenRef.current) {
        retryRef.current = setTimeout(reconnect, 1200);
      }
    };

    ws.onerror = () => { if (socketRef.current === ws) setStatus("error"); };
  }, [matchId]);

  const start = useCallback(async () => {
    wantOpenRef.current = true;
    seqRef.current = 0;
    closeSocket();
    setStatus("connecting");
    try {
      await api.startLive(matchId, 90);
      if (wantOpenRef.current) connect();
    } catch {
      wantOpenRef.current = false;
      closeSocket();
      setStatus("error");
    }
  }, [connect, closeSocket, matchId]);

  const stop = useCallback(async () => {
    wantOpenRef.current = false;
    try { await api.stopLive(matchId); } catch { /* stopping is best-effort */ }
    closeSocket();
    setStatus("idle");
  }, [closeSocket, matchId]);

  useEffect(() => () => { wantOpenRef.current = false; closeSocket(); }, [closeSocket]);

  // Auto-start once on mount when the page is the dedicated live view.
  const startedRef = useRef(false);
  useEffect(() => {
    if (autoStart && !startedRef.current) {
      startedRef.current = true;
      void start();
    }
  }, [autoStart, start]);

  // Advance the seconds hand between server clock ticks so the display reads
  // like a running clock rather than a value that jumps once a minute.
  useEffect(() => {
    if (status !== "live") return;
    const id = setInterval(() => setSecond((s) => (s + 1) % 60), 1000);
    return () => clearInterval(id);
  }, [status]);

  return { status, connected, minute, second, homeScore, awayScore, events, momentum, start, stop };
}
