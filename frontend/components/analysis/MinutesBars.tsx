"use client";

import { useState } from "react";
import { summarize } from "@/lib/insights.mjs";
import type { ChartRecord } from "@/lib/types";

const shortDate = (date: string) => `${Number(date.slice(5, 7))}/${date.slice(8, 10)}`;
const label = (minutes: number) => (Number.isInteger(minutes) ? String(minutes) : minutes.toFixed(1));

/** Minutes played in each game of the chosen window: the context behind any counting stat. */
export function MinutesBars({ games }: { games: ChartRecord[] }) {
  const [mode, setMode] = useState<"average" | "median">("average");
  const rows = games.filter((game): game is ChartRecord & { minutes: number } => game.minutes != null);
  const summary = summarize(rows.map(game => game.minutes));
  if (!summary) return null;
  const top = Math.max(...rows.map(game => game.minutes), 1);
  const headline = mode === "average" ? summary.mean : summary.median;

  return (
    <div className="mt-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="ql-data-label">MINUTES</p>
          <p className="mt-1 text-2xl font-bold tabular-nums text-text-pri">{headline.toFixed(1)}</p>
        </div>
        <div className="flex gap-1" role="group" aria-label="Minutes summary">
          {(["average", "median"] as const).map(option => (
            <button key={option} type="button" aria-pressed={mode === option} onClick={() => setMode(option)}
              className={`ql-chip min-h-8 px-3 py-1 text-xs font-semibold capitalize focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8] ${mode === option ? "ql-chip-active" : "text-[#a9bec0]"}`}>{option}</button>
          ))}
        </div>
      </div>
      <div className="mt-3 overflow-x-auto">
        <ul className="flex h-40 min-w-[320px] items-end gap-1.5" aria-label="Minutes per game">
          {rows.map((game, index) => (
            <li key={`${game.date}-${index}`} className="flex h-full min-w-0 flex-1 flex-col justify-end text-center">
              <span className="text-[10px] font-semibold tabular-nums text-text-sec">{label(game.minutes)}</span>
              <span className="mt-1 block rounded-t bg-white/15" style={{ height: `${Math.max(4, (game.minutes / top) * 100)}%` }}
                title={`${game.date} · ${game.opponent} · ${label(game.minutes)} min`} />
              <span className="mt-1 text-[10px] tabular-nums text-text-sec">{shortDate(game.date)}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
