"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

const labels: Record<string, string> = { PTS: "Points", REB: "Rebounds", AST: "Assists", FG3M: "Threes made" };

export default function HitRatesPage() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["wnba-hitrates"], queryFn: api.hitrates, staleTime: 120_000 });

  return <div className="ql-page max-w-[1260px]">
    <header className="mb-6 border-b border-[#2b4248] pb-5">
      <p className="ql-kicker">05 / WNBA research</p>
      <h1 className="ql-heading mt-2">WNBA Hit Rates</h1>
      <p className="ql-subtitle mt-2 max-w-3xl">Meaningful thresholds cleared in at least eight of the last ten games by players in today&apos;s matchups. Historical results are not a sportsbook pick or a forecast.</p>
    </header>

    {isPending && <div className="ql-panel h-52 animate-pulse" aria-label="Loading hit rates" />}
    {isError && <div role="alert" className="ql-panel p-5 text-sm text-red-300">Hit rates are unavailable. <button onClick={() => void refetch()} className="underline">Try again</button></div>}
    {data?.message && <p role="status" className="ql-panel p-6 text-sm text-text-sec">{data.message}</p>}
    {!isPending && !isError && data?.games.length === 0 && !data.message && <p role="status" className="ql-panel p-6 text-sm text-text-sec">No qualifying hit-rate thresholds are available for the current WNBA slate.</p>}

    <div className="space-y-5">{data?.games.map(game => <section key={game.matchup} className="ql-panel overflow-hidden" aria-label={game.matchup}>
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[#2b4248] bg-[#1a2930] px-5 py-4">
        <h2 className="ql-section-title">{game.matchup}</h2>
        <span className="ql-data-label">{game.total_count} qualifying thresholds</span>
      </div>
      <div className="divide-y divide-[#2b4248]">{game.entries.map((entry, index) => <div key={`${entry.player_name}-${entry.stat}-${entry.threshold}-${index}`} className="grid items-center gap-3 px-5 py-3.5 sm:grid-cols-[54px_minmax(160px,1fr)_minmax(100px,2fr)_70px]">
        <span className="rounded border border-[#2b4248] px-2 py-1 text-center font-mono text-[10px] font-bold text-teal-300">{entry.team}</span>
        <div className="min-w-0"><Link href={`/analysis?league=wnba&player=${encodeURIComponent(entry.player_name)}`} className="font-semibold text-text-pri hover:text-teal-300">{entry.player_name}</Link><p className="text-xs text-text-sec"><strong className="text-teal-300">{entry.threshold}+</strong> {labels[entry.stat] || entry.stat}</p></div>
        <div className="hidden h-2 rounded-sm bg-[#20363a] sm:block" role="img" aria-label={`${entry.hits} hits in ${entry.games} games`}><span className="block h-full rounded-sm bg-teal-400" style={{ width: `${entry.games ? Math.min(100, entry.hits / entry.games * 100) : 0}%` }} /></div>
        <strong className="font-mono text-lg tabular-nums text-teal-300 sm:text-right">{entry.hits}/{entry.games}</strong>
      </div>)}</div>
    </section>)}</div>
  </div>;
}
