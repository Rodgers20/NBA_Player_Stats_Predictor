"use client";

import { useEffect, useRef } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { findPlayer, liveBet, nameKey, statValue, type LiveBox, type LiveGame } from "@/lib/espn-live.mjs";
import { easternToday, easternYesterday, summaryQuery, useLiveScoreboard } from "@/lib/live";
import type { Bet, Result } from "@/lib/journal-client";

type Final = (bet: Bet, result: Result) => void;
const TONE = { win: "border-green-400/40 text-green-300", loss: "border-red-400/40 text-red-300", push: "border-ink/20 text-text-sec", live: "border-teal-400/40 text-teal-300" };

/** Pending bets from today (and last night's late games) graded against ESPN's live box score. */
export function LiveBets({ bets, onFinal }: { bets: Bet[]; onFinal: Final }) {
  const dates = [easternToday(), easternYesterday()];
  const open = bets.filter(bet => bet.result === "pending" && dates.includes(bet.game_date));
  if (!open.length) return null;
  const groups = [...new Set(open.map(bet => `${bet.league}|${bet.game_date}`))];
  return <section className="ql-panel p-5 md:p-6" aria-labelledby="live-bets-title">
    <p className="ql-kicker flex items-center gap-2"><i aria-hidden className="h-2 w-2 animate-pulse rounded-full bg-red-500" /> Live tracker</p>
    <h2 id="live-bets-title" className="ql-section-title mt-2">Your open bets, live</h2>
    <p className="mt-2 text-sm text-[var(--color-text-sec)]">Updates from ESPN every 15 seconds while a game is on. Bets grade themselves at the final; correct any your book settled differently.</p>
    <ul className="mt-4 divide-y divide-border-card">{groups.map(group => {
      const [league, date] = group.split("|");
      return <LeagueDay key={group} league={league} date={date} bets={open.filter(bet => `${bet.league}|${bet.game_date}` === group)} onFinal={onFinal} />;
    })}</ul>
  </section>;
}

function LeagueDay({ league, date, bets, onFinal }: { league: string; date: string; bets: Bet[]; onFinal: Final }) {
  const board = useLiveScoreboard(league, date);
  const games = useQuery({ queryKey: ["games", league, date], queryFn: () => api.games(league, date), staleTime: 300_000 });
  // The player's team (from their profile) finds their game, so only that game's feed is polled.
  const players = [...new Set(bets.map(bet => bet.player))];
  const profiles = useQueries({ queries: players.map(name => ({ queryKey: ["stats", name, league], queryFn: () => api.playerStats(name, league), staleTime: 3_600_000, retry: false })) });
  // The day's roster wins: some duplicate profiles (unaccented names) carry no team.
  const roster = useQuery({ queryKey: ["scheduled-players", league, date], queryFn: () => api.scheduledPlayers(league, date), staleTime: 300_000 });
  const rosterTeam = new Map((roster.data?.players ?? []).map(p => [nameKey(p.name), p.team]));
  const teamOf = new Map(players.map((name, i) => [name, rosterTeam.get(nameKey(name)) || profiles[i].data?.team]));
  const eventOf = (bet: Bet) => {
    const team = teamOf.get(bet.player);
    return (team && games.data?.games.find(g => g.home_team === team || g.away_team === team)?.game_id) || null;
  };
  // Only games that have tipped off are fetched.
  // ponytail: a player with no known team scans every started game of that date (finals load once, never poll).
  const started = Object.values(board).filter(game => game.state !== "pre").map(game => game.id);
  const mapped = bets.map(eventOf);
  const ids = [...new Set(mapped.some(id => !id) ? started : mapped.filter((id): id is string => !!id && started.includes(id)))];
  const summaries = useQueries({ queries: ids.map(id => summaryQuery(league, id)) });
  const boxes = summaries.map(s => s.data).filter((box): box is LiveBox => !!box);
  const rows = bets.map((bet, i) => {
    const event = mapped[i];
    const candidates = event ? boxes.filter(box => box.game_id === event) : boxes;
    const box = candidates.find(item => findPlayer(item, bet.player)) ?? (event ? candidates[0] : undefined);
    const player = box ? findPlayer(box, bet.player) : null;
    const value = player ? statValue(player.stats, bet.stat) : null;
    return { bet, box, game: board[event ?? box?.game_id ?? ""], grade: box && value != null ? liveBet(bet, value, box.state) : null };
  });

  const graded = useRef(new Set<string>());
  useEffect(() => {
    for (const { bet, box, grade } of rows) {
      if (box?.state !== "post" || !grade || grade.status === "live" || graded.current.has(bet.id)) continue;
      graded.current.add(bet.id);
      onFinal(bet, grade.status);
    }
  });

  return <>{rows.map(row => <LiveRow key={row.bet.id} {...row} />)}</>;
}

function LiveRow({ bet, box, game, grade }: { bet: Bet; box?: LiveBox; game?: LiveGame; grade: ReturnType<typeof liveBet> | null }) {
  const over = bet.side !== "Under";
  const share = grade ? Math.min(1, grade.value / Math.max(bet.line, 0.5)) : 0;
  const clock = box ? box.status : game?.detail ?? "Waiting for the schedule";
  const note = !grade ? (box ? "Not in the box score yet" : "Tracks from tip-off")
    : grade.status !== "live" ? { win: "Hit", loss: "Missed", push: "Push" }[grade.status]
      : over ? `Needs ${grade.need} more` : `${Math.max(0, Math.ceil(bet.line) - 1 - grade.value)} to spare`;
  return <li className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-2 py-3">
    <div className="min-w-0">
      <p className="truncate font-semibold">{bet.player}</p>
      <p className="text-xs text-[var(--color-text-sec)]">{bet.side} {bet.line} {bet.stat} · {bet.league.toUpperCase()} · {clock}</p>
    </div>
    <div className="flex items-center gap-3">
      <span className="font-mono text-2xl font-bold tabular-nums" aria-label={`${grade?.value ?? 0} of ${bet.line}`}>{grade?.value ?? "—"}<small className="text-xs text-[var(--color-text-sec)]"> / {bet.line}</small></span>
      <span role="status" className={`rounded border px-2 py-1 text-xs font-semibold ${TONE[grade?.status ?? "live"]}`}>{note}</span>
    </div>
    <div className="col-span-2 h-1.5 overflow-hidden rounded bg-border-card" aria-hidden>
      <div className={`h-full origin-left transition-transform duration-500 ${grade?.status === "loss" ? "bg-red-400" : grade?.status === "win" ? "bg-green-400" : "bg-teal-400"}`} style={{ transform: `scaleX(${share})` }} />
    </div>
  </li>;
}
