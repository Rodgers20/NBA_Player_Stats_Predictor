"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Prop } from "@/lib/types";
import { PlayerBarChart } from "@/components/charts/PlayerBarChart";
import { gradeSide, hitSummary } from "@/lib/hit-rate.mjs";
import { bookSeries, movement } from "@/lib/line-movement.mjs";

const STATS = ["PTS", "REB", "AST", "FG3M", "PTS+REB", "PTS+AST", "REB+AST", "PTS+REB+AST"];

export function PropAnalysisPanel({ prop, league, onClose }: { prop: Prop; league: "nba" | "wnba"; onClose: () => void }) {
  const [stat, setStat] = useState(prop.stat);
  const [photoFailed, setPhotoFailed] = useState(false);
  const [span, setSpan] = useState(10);
  const [tab, setTab] = useState<"games" | "movement">("games");
  const chart = useQuery({
    queryKey: ["prop-preview", league, prop.player, stat, span],
    queryFn: () => api.playerChart(prop.player, stat, span, league),
  });
  const history = useQuery({
    queryKey: ["line-history", league, prop.player, prop.stat],
    queryFn: () => api.lineHistory(prop.player, prop.stat, league),
    enabled: tab === "movement",
  });
  const snapshots = history.data?.snapshots ?? [];
  const lineBooks = [...bookSeries(snapshots)].map(([book, points]) => ({ book, move: movement(points) }));
  const games = chart.data?.games ?? [];
  const completedGames = games.filter(game => game.value != null);
  const line = stat === prop.stat ? prop.line : null;
  const side = prop.direction.toLowerCase() === "under" ? "Under" : "Over";
  const hits = line == null ? null : hitSummary(completedGames.map(game => game.value as number), line, side).hits;
  const chartGames = games.map(game => ({
    ...game,
    hit: line == null || game.value == null ? null : gradeSide(game.value, line, side) === "hit",
  }));

  return <aside className="ql-panel fixed inset-x-3 bottom-3 top-16 z-40 overflow-y-auto p-4 shadow-xl lg:left-auto lg:right-4 lg:w-[420px] 2xl:sticky 2xl:inset-auto 2xl:top-24 2xl:z-auto 2xl:w-auto 2xl:max-h-[calc(100vh-7rem)]" aria-label={`${prop.player} prop history`}>
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

    <div className="mt-5 flex gap-1 border-b border-white/10" role="group" aria-label="Inspector view">
      {([["games", "Game log"], ["movement", "Line movement"]] as const).map(([value, label]) => <button key={value} type="button" aria-pressed={tab === value} onClick={() => setTab(value)}
        className={`border-b-2 px-3 py-2 text-xs font-bold uppercase tracking-[.08em] ${tab === value ? "border-[#59e0c8] text-[#59e0c8]" : "border-transparent text-[#a9bec0] hover:text-white"}`}>{label}</button>)}
    </div>

    {tab === "movement" && <div className="mt-4">
      {history.isPending && <div className="h-24 animate-pulse rounded-lg bg-white/5" />}
      {history.isError && <p role="alert" className="text-sm text-red-300">Line history is unavailable. <button type="button" onClick={() => void history.refetch()} className="underline">Try again</button></p>}
      {!history.isPending && !history.isError && lineBooks.length === 0 && <p className="text-sm text-[#a9bec0]">No history yet; snapshots are saved on each odds refresh</p>}
      {lineBooks.length > 0 && <table className="w-full border-collapse text-xs text-[#a9bec0]">
        <thead><tr className="ql-data-label text-[10px]"><th className="px-2 py-1 text-left font-normal">BOOK</th><th className="px-2 py-1 text-right font-normal">OPEN</th><th className="px-2 py-1 text-right font-normal">NOW</th><th className="px-2 py-1 text-right font-normal">MOVE</th><th className="px-2 py-1 text-right font-normal">SNAPS</th></tr></thead>
        <tbody>{lineBooks.map(({ book, move }) => move && <tr key={book} className="border-t border-white/[0.06] tabular-nums">
          <td className="px-2 py-1.5 text-[#eaf8f6]">{book}</td><td className="px-2 py-1.5 text-right">{move.opening}</td><td className="px-2 py-1.5 text-right text-[#eaf8f6]">{move.current}</td>
          <td className={`px-2 py-1.5 text-right ${move.change === 0 ? "" : "font-bold text-[#f5ba64]"}`}>{move.change > 0 ? "+" : ""}{move.change}</td><td className="px-2 py-1.5 text-right">{move.samples}</td>
        </tr>)}</tbody>
      </table>}
    </div>}

    {tab === "games" && <div className="mt-5 pt-1">
      <div className="mb-3 flex gap-1.5" role="group" aria-label="Game window">
        {[5, 10, 20].map(n => <button key={n} type="button" aria-pressed={span === n} onClick={() => setSpan(n)} className={`ql-chip px-2.5 py-1 text-xs ${span === n ? "ql-chip-active" : "text-[#a9bec0] hover:text-[#eaf8f6]"}`}>L{n}</button>)}
      </div>
      <div className="flex flex-wrap items-baseline justify-between gap-2"><p className="ql-section-title">Last {span} games · {stat}</p><p className="ql-data-label text-[#a9bec0]">AVG <strong className="text-[#eaf8f6]">{chart.data?.avg?.toFixed(1) ?? "—"}</strong></p></div>
      {line != null && <p className="mt-2 text-xs text-[#59e0c8]"><strong className="text-base tabular-nums">{hits ?? 0}/{completedGames.length}</strong> historical hits {prop.direction.toLowerCase()} {line}</p>}
      {chart.isPending && <div className="mt-3 h-48 animate-pulse rounded-lg bg-white/5" />}
      {chart.isError && <p role="alert" className="mt-3 text-sm text-red-300">Game history is unavailable. <button type="button" onClick={() => void chart.refetch()} className="underline">Try again</button></p>}
      {!chart.isPending && !chart.isError && games.length === 0 && <p className="mt-3 text-sm text-[#a9bec0]">No completed games are available.</p>}
      {games.length > 0 && <div className="mt-3 overflow-x-auto"><PlayerBarChart games={chartGames} stat={stat} line={line} height={210} /></div>}
    </div>}

    <Link href={`/analysis?league=${league}&player=${encodeURIComponent(prop.player)}`} className="ql-control mt-4 inline-flex px-4 py-2.5 text-sm font-bold text-[#59e0c8] hover:bg-[#59e0c8]/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">Full player analysis →</Link>
  </aside>;
}
