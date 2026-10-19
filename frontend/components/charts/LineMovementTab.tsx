"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { bookSeries, movement, primaryBook } from "@/lib/line-movement.mjs";
import { fmtOdds } from "@/lib/utils";
import { LineMovementChart } from "./LineMovementChart";

export function LineMovementTab({ player, stat, league }: { player: string; stat: string; league: "nba" | "wnba" }) {
  const [chosen, setChosen] = useState<string | null>(null);
  const history = useQuery({ queryKey: ["line-history", league, player, stat], queryFn: () => api.lineHistory(player, stat, league), staleTime: 60_000 });
  const snapshots = history.data?.snapshots ?? [];
  const series = bookSeries(snapshots);
  const book = chosen && series.has(chosen) ? chosen : primaryBook(snapshots);
  const points = book ? series.get(book) ?? [] : [];
  const change = movement(points);
  const latest = points[points.length - 1];

  if (history.isPending) return <div className="mt-3 h-40 animate-pulse rounded-lg bg-white/5" />;
  if (history.isError) return <p role="alert" className="mt-3 text-sm text-red-300">Line history is unavailable. <button type="button" onClick={() => void history.refetch()} className="underline">Try again</button></p>;
  if (!book || !change) return <p role="status" className="mt-3 text-sm text-[#a9bec0]">No history yet; snapshots are saved on each odds refresh.</p>;

  return <div className="mt-3">
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Sportsbook">
      {[...series.keys()].map(name => <button key={name} type="button" aria-pressed={book === name} onClick={() => setChosen(name)} className={`ql-chip px-2.5 py-1 text-xs font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8] ${book === name ? "ql-chip-active" : "text-[#a9bec0] hover:text-[#eaf8f6]"}`}>{name}</button>)}
    </div>
    <p className="mt-3 text-xs text-[#a9bec0]">
      Opened <strong className="tabular-nums text-[#eaf8f6]">{change.opening}</strong> → now <strong className="tabular-nums text-[#eaf8f6]">{change.current}</strong>
      {change.change !== 0 && <span className={change.change > 0 ? "text-[#f5ba64]" : "text-[#59e0c8]"}> ({change.change > 0 ? "+" : ""}{change.change})</span>}
      {" "}· {change.samples} snapshot{change.samples === 1 ? "" : "s"}
    </p>
    <div className="mt-2"><LineMovementChart points={points} book={book} /></div>
    {latest && <p className="ql-data-label mt-1 text-[#a9bec0]">LATEST · OVER {fmtOdds(latest.over_price)} · UNDER {fmtOdds(latest.under_price)}</p>}
  </div>;
}
