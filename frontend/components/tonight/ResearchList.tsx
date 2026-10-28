"use client";

import { useState } from "react";
import Link from "next/link";
import type { Prop } from "@/lib/types";
import { PlayerName, focusFromProp } from "@/components/analysis/PlayerName";

const SHOWN = 15;
const fmt = (value: number | null | undefined) => (value == null ? "—" : value.toFixed(1));

/** Tonight's model projections for games with no sportsbook line yet. Nothing here is a bet or carries an edge. */
export function ResearchList({ rows, league }: { rows: Prop[]; league: "nba" | "wnba" }) {
  const stats = [...new Set(rows.map(row => row.stat))];
  const [chosen, setChosen] = useState(stats[0] ?? "");
  const stat = stats.includes(chosen) ? chosen : stats[0];
  const shown = rows.filter(row => row.stat === stat)
    .sort((a, b) => (b.model_projection ?? -Infinity) - (a.model_projection ?? -Infinity)).slice(0, SHOWN);

  return <section className="mt-8" aria-label="Model projections without a line">
    <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2 border-b border-ink/10 pb-2">
      <h2 className="ql-section-title">Model projections · no line yet</h2>
      <div className="flex gap-1.5" role="group" aria-label="Projection stat">
        {stats.map(value => <button key={value} type="button" aria-pressed={value === stat} onClick={() => setChosen(value)}
          className={`ql-chip transition-[transform,background-color,color] duration-150 ease-out active:scale-[0.97] ${value === stat ? "ql-chip-active" : ""}`}>{value}</button>)}
      </div>
    </div>
    <p className="mb-3 text-xs text-text-sec">Not bets. No sportsbook has posted a line, so no edge is claimed and these can&apos;t go on your slip. They turn into plays above once a fresh price exists.</p>
    <div className="ql-panel overflow-hidden">
      {shown.map(row => <div key={`${row.player}-${row.stat}`} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-b border-ink/[0.07] px-4 py-3 last:border-b-0 sm:px-5">
        <div className="min-w-0">
          <PlayerName name={row.player} league={league} focus={focusFromProp(row)} className="block max-w-full truncate text-sm font-bold text-text-pri" />
          <p className="truncate text-xs text-text-sec">{row.game_matchup || `${row.team} vs ${row.opponent}`}</p>
        </div>
        <div className="text-right">
          <p className="text-lg font-extrabold tabular-nums text-teal-400">{fmt(row.model_projection)} <span className="text-xs font-bold text-text-sec">{row.stat_label || row.stat}</span></p>
          <p className="text-[11px] tabular-nums text-text-sec">L5 {fmt(row.l5_avg)} · avg {fmt(row.avg)}</p>
        </div>
      </div>)}
    </div>
    <Link href={`/analysis?league=${league}`} className="mt-3 inline-block text-xs font-bold text-teal-400">Dig into any player →</Link>
  </section>;
}
