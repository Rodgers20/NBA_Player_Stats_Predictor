"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import { fmtOdds } from "@/lib/utils";
import { formatOdds } from "@/lib/prop-research.mjs";
import type { TrendCard } from "@/lib/types";

const SIDES = [["all", "All"], ["over", "Overs"], ["under", "Unders"]] as const;
const STATS = [["all", "All"], ["PTS", "PTS"], ["REB", "REB"], ["AST", "AST"], ["FG3M", "3PM"]] as const;
const focus = "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]";

function Stars({ count }: { count: number }) {
  return <span role="img" aria-label={`${count} of 5 stars`} className="text-sm tracking-tight text-[#f5ba64]">{"★".repeat(count)}<span className="text-white/15">{"★".repeat(5 - count)}</span></span>;
}

function TrendItem({ card, league }: { card: TrendCard; league: string }) {
  const stat = card.stat === "FG3M" ? "3PM" : card.stat;
  return <li className="ql-panel flex flex-col gap-3 p-4">
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <Link href={`/analysis?league=${league}&player=${encodeURIComponent(card.player)}`} className={`block truncate font-bold text-text-pri hover:text-teal-300 ${focus}`}>{card.player}</Link>
        <p className="ql-data-label mt-0.5">{card.team}{card.opponent ? ` vs ${card.opponent}` : ""} · {stat}</p>
      </div>
      <Stars count={card.stars} />
    </div>
    <p className="text-base font-extrabold leading-snug text-text-pri">{card.text}</p>
    <div className="flex gap-1" role="img" aria-label={`Last ${card.games} games, oldest to newest: ${card.results.map(hit => hit ? "hit" : "miss").join(", ")}`}>
      {card.results.map((hit, index) => <span key={index} className={`h-5 flex-1 rounded-sm ${hit ? "bg-teal-400" : "bg-rose-400/70"}`} />)}
    </div>
    <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-text-sec">
      <span className="tabular-nums"><strong className="text-sm text-teal-400">{card.hits}/{card.games}</strong> hit · avg {card.avg}</span>
      {card.live
        ? <span className="ql-data-label rounded border border-teal-400/30 px-2 py-1 text-teal-300">LIVE {card.live.line} {formatOdds(card.live.price) ?? fmtOdds(card.live.price)}{card.live.book ? ` · ${card.live.book}` : ""}</span>
        : <span className="ql-data-label">DERIVED LINE {card.line}</span>}
    </div>
  </li>;
}

export default function TrendsPage() {
  const league = usePrefs(state => state.league);
  const [side, setSide] = useState<(typeof SIDES)[number][0]>("all");
  const [stat, setStat] = useState<(typeof STATS)[number][0]>("all");
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["trends", league], queryFn: () => api.trends(league, { limit: 200 }), staleTime: 300_000 });
  const cards = (data?.trends ?? []).filter(card => (side === "all" || card.side === side) && (stat === "all" || card.stat === stat));
  const chip = (active: boolean) => `ql-chip shrink-0 ${active ? "ql-chip-active" : ""}`;

  return <div className="ql-page mx-auto max-w-[1700px] px-4 py-6 sm:px-8">
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <p className="ql-kicker">{league.toUpperCase()} / STREAK FINDER</p>
        <h1 className="ql-heading mt-1">Trends</h1>
        <p className="ql-subtitle mt-1 max-w-3xl">Players clearing the same line in at least eight of their last ten games. Stars grow with the hit count. History is not a forecast.</p>
      </div>
      <span className="ql-data-label border border-white/10 px-3 py-2">{data ? `${cards.length} OF ${data.trends.length} STREAKS` : `${league.toUpperCase()} / TRENDS`}</span>
    </div>

    <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-y border-white/10 py-3">
      <div className="flex max-w-full gap-1.5 overflow-x-auto pb-1" role="group" aria-label="Side">{SIDES.map(([value, label]) => <button key={value} type="button" aria-pressed={side === value} onClick={() => setSide(value)} className={chip(side === value)}>{label}</button>)}</div>
      <div className="flex max-w-full gap-1.5 overflow-x-auto pb-1" role="group" aria-label="Statistic">{STATS.map(([value, label]) => <button key={value} type="button" aria-pressed={stat === value} onClick={() => setStat(value)} className={chip(stat === value)}>{label}</button>)}</div>
    </div>

    {isPending && <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" aria-label="Loading trends">{[0, 1, 2, 3].map(index => <div key={index} className="ql-panel h-40 animate-pulse" />)}</div>}
    {isError && <div role="alert" className="ql-panel p-5 text-sm text-red-300">Trends are unavailable. <button type="button" onClick={() => void refetch()} className="underline">Try again</button></div>}
    {!isPending && !isError && cards.length === 0 && <div role="status" className="ql-panel p-6">
      <h2 className="ql-section-title">{data?.trends.length ? "No trends match these filters" : "No qualifying streaks right now"}</h2>
      <p className="mt-2 max-w-2xl text-sm text-text-sec">{data?.trends.length ? "Try another side or stat." : "A trend needs a player with ten games inside the last 14 days, so this page fills in once the season is underway and goes quiet in the off-season."}</p>
    </div>}
    <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">{cards.map(card => <TrendItem key={`${card.player}-${card.stat}-${card.side}`} card={card} league={league} />)}</ul>
  </div>;
}
