"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Prop } from "@/lib/types";
import { PlayerBarChart } from "@/components/charts/PlayerBarChart";

const STATS = ["PTS", "REB", "AST", "FG3M", "PTS+REB", "PTS+AST", "REB+AST", "PTS+REB+AST"];

export function PropAnalysisPanel({ prop, league, onClose }: { prop: Prop; league: "nba" | "wnba"; onClose: () => void }) {
  const [stat, setStat] = useState(prop.stat);
  const [photoFailed, setPhotoFailed] = useState(false);
  const chart = useQuery({
    queryKey: ["prop-preview", league, prop.player, stat],
    queryFn: () => api.playerChart(prop.player, stat, 10, league),
  });
  const games = chart.data?.games ?? [];
  const completedGames = games.filter(game => game.value != null);
  const line = stat === prop.stat ? prop.line : null;
  const hits = line == null ? null : games.filter(game => game.value != null && (prop.direction.toLowerCase() === "under" ? game.value < line : game.value > line)).length;
  const chartGames = games.map(game => ({
    ...game,
    hit: line == null || game.value == null ? null : prop.direction.toLowerCase() === "under" ? game.value < line : game.value > line,
  }));

  return <aside className="ql-panel fixed inset-x-3 bottom-3 top-16 z-40 overflow-y-auto p-4 shadow-xl lg:sticky lg:inset-auto lg:top-24 lg:z-auto lg:max-h-[calc(100vh-7rem)]" aria-label={`${prop.player} prop history`}>
    <div className="flex items-start gap-3">
      {prop.headshot_url && !photoFailed
        ? <Image src={prop.headshot_url} alt="" width={56} height={56} unoptimized onError={() => setPhotoFailed(true)} className="h-14 w-14 rounded-md border border-white/10 object-cover object-top" />
        : <div className="flex h-14 w-14 items-center justify-center rounded-md border border-white/10 bg-[#59e0c8]/10 text-lg font-bold text-[#59e0c8]" aria-hidden="true">{prop.player.trim()[0]}</div>}
      <div className="min-w-0 flex-1"><p className="ql-kicker">PLAYER INSPECTOR</p><h2 className="truncate text-xl font-extrabold text-[#eaf8f6]">{prop.player}</h2><p className="text-xs text-[#a9bec0]">{prop.team} · {prop.game_matchup || prop.opponent}</p></div>
      <button type="button" onClick={onClose} aria-label="Close player history" className="ql-control px-2.5 py-1.5 text-sm text-[#d0e0df] hover:text-[#eaf8f6] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">✕</button>
    </div>

    <div className="mt-5 flex flex-wrap gap-1.5" role="group" aria-label="Preview statistic">
      {[...new Set([prop.stat, ...STATS])].map(value => <button key={value} type="button" aria-pressed={stat === value} onClick={() => setStat(value)} className={`ql-chip px-2.5 py-1.5 text-xs font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8] ${stat === value ? "ql-chip-active" : "text-[#a9bec0] hover:text-[#eaf8f6]"}`}>{value === "FG3M" ? "3PM" : value}</button>)}
    </div>

    <div className="mt-5 border-t border-white/10 pt-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2"><p className="ql-section-title">Last 10 games · {stat}</p><p className="ql-data-label text-[#a9bec0]">AVG <strong className="text-[#eaf8f6]">{chart.data?.avg?.toFixed(1) ?? "—"}</strong></p></div>
      {line != null && <p className="mt-2 text-xs text-[#59e0c8]"><strong className="text-base tabular-nums">{hits ?? 0}/{completedGames.length}</strong> historical hits {prop.direction.toLowerCase()} {line}</p>}
      {chart.isPending && <div className="mt-3 h-48 animate-pulse rounded-lg bg-white/5" />}
      {chart.isError && <p role="alert" className="mt-3 text-sm text-red-300">Game history is unavailable. <button type="button" onClick={() => void chart.refetch()} className="underline">Try again</button></p>}
      {!chart.isPending && !chart.isError && games.length === 0 && <p className="mt-3 text-sm text-[#a9bec0]">No completed games are available.</p>}
      {games.length > 0 && <div className="mt-3 overflow-x-auto"><PlayerBarChart games={chartGames} stat={stat} line={line} height={210} /></div>}
    </div>

    <Link href={`/analysis?league=${league}&player=${encodeURIComponent(prop.player)}`} className="ql-control mt-4 inline-flex px-4 py-2.5 text-sm font-bold text-[#59e0c8] hover:bg-[#59e0c8]/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">Full player analysis →</Link>
  </aside>;
}
