"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { hitChips } from "@/lib/prop-research.mjs";
import type { Prop } from "@/lib/types";
import { LineMovementTab } from "@/components/charts/LineMovementTab";

const card = "ql-panel p-4 sm:p-5";
const TONE = { good: "text-teal-400", bad: "text-rose-400", neutral: "text-text-pri" } as const;
const one = (value: number) => value.toFixed(1);

/** The player's posted market for the selected stat, if the board has one. */
function usePostedProp(league: "nba" | "wnba", player: string, stat: string): Prop | undefined {
  const board = useQuery({
    queryKey: ["props", "analysis-lookup", league],
    queryFn: () => api.props({ league, include_research: true, sort: "ev", limit: 500 }),
    staleTime: 120_000,
  });
  const rows = (board.data?.props ?? []).filter(row => row.player === player && row.stat === stat);
  return rows.find(row => row.teammate_splits?.length) ?? rows[0];
}

export function PlayerResearch({ league, player, stat }: { league: "nba" | "wnba"; player: string; stat: string }) {
  const prop = usePostedProp(league, player, stat);
  const splits = prop?.teammate_splits ?? [];
  const chips = hitChips(prop?.hit_windows);
  const side = prop?.direction?.toLowerCase() === "under" ? "Under" : "Over";

  return <>
    {splits.length > 0 && <section className={card}>
      <p className="ql-kicker">SUPPORTING EVIDENCE</p>
      <h2 className="ql-section-title mt-1">Teammate Out</h2>
      <p className="mb-3 mt-1 text-xs text-text-sec">{stat} splits when a rotation teammate is unavailable, graded against the posted line.</p>
      <div className="divide-y divide-ink/10">{splits.map(item => {
        const delta = item.avg_without - item.avg_with;
        return <div key={item.teammate} className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 py-3 text-sm">
          <p className="text-text-sec">Without <strong className="text-text-pri">{item.teammate}</strong> <span className="font-mono text-[11px] text-amber-300">{item.status}</span></p>
          <p className="text-text-sec">
            avg <strong className={`tabular-nums ${delta >= 0 ? "text-teal-400" : "text-rose-400"}`}>{one(item.avg_without)}</strong> vs {one(item.avg_with)} with
            {item.hit_rate_without != null && <> · hit rate <strong className="tabular-nums text-text-pri">{Math.round(item.hit_rate_without)}%</strong></>} over {item.games_without} game{item.games_without === 1 ? "" : "s"}
          </p>
        </div>;
      })}</div>
    </section>}

    <section className={card}>
      <p className="ql-kicker">MARKET</p>
      <h2 className="ql-section-title mt-1">Line Movement</h2>
      {prop && chips.length > 0 && <div className="mt-3">
        <p className="ql-data-label">{side.toUpperCase()} {prop.line} HIT RATE</p>
        <div className="mt-2 flex flex-wrap gap-1.5">{chips.map(chip => <span key={chip.key} className="ql-chip min-h-0 px-2.5 py-1 text-xs">
          {chip.label} <strong className={`tabular-nums ${TONE[chip.tone]}`}>{chip.hits}/{chip.total}</strong></span>)}</div>
      </div>}
      <LineMovementTab player={player} stat={stat} league={league} />
    </section>
  </>;
}
