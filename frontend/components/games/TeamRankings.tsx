"use client";

import { useState } from "react";
import { ArrowLeftRight } from "lucide-react";
import { offenseVsDefense, type RankCell } from "@/lib/team-rankings.mjs";
import type { MatchupProfile } from "@/lib/types";

/** How strong a rank is, 0 to 1: 1st is 1, last is close to 0. */
const strength = ({ rank, of }: RankCell) => (of - rank + 1) / of;

function Side({ cell, align, lead }: { cell: RankCell; align: "left" | "right"; lead: boolean }) {
  return (
    <p className={`min-w-0 tabular-nums ${align === "right" ? "text-right" : ""}`}>
      <strong className={`text-base ${lead ? "text-teal-300" : "text-text-pri"}`}>{cell.display}</strong>
      <span className="ml-2 text-[11px] text-text-sec">#{cell.rank}</span>
    </p>
  );
}

/** One team's offense against the other's defense. Rank 1 is best on both sides; swap to read it the other way. */
export function TeamRankings({ away, home, profiles, season }: {
  away: string; home: string; profiles: Record<string, MatchupProfile | null | undefined> | undefined; season: string | null;
}) {
  const [flipped, setFlipped] = useState(false);
  const offenseTeam = flipped ? home : away;
  const defenseTeam = flipped ? away : home;
  const rows = offenseVsDefense(profiles, offenseTeam, defenseTeam);
  if (!rows.length) return null;

  return (
    <section aria-label="Team rankings">
      <h3 className="ql-data-label mb-2">TEAM RANKINGS{season ? ` (${season})` : ""}</h3>
      <div className="ql-panel p-4">
        <div className="flex items-center justify-between gap-3 text-sm font-bold text-text-pri">
          <span>{offenseTeam} offense</span>
          <button type="button" onClick={() => setFlipped(value => !value)} aria-label={`Swap: show ${defenseTeam} offense against ${offenseTeam} defense`}
            className="ql-control grid h-8 w-8 place-items-center text-text-sec transition-transform duration-150 active:scale-95 hover:text-text-pri focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">
            <ArrowLeftRight size={15} aria-hidden="true" />
          </button>
          <span>{defenseTeam} defense</span>
        </div>
        <ul className="mt-3 divide-y divide-white/[0.07]">
          {rows.map(row => {
            const left = strength(row.offense), right = strength(row.defense);
            return (
              <li key={row.label} className="py-3">
                <div className="grid grid-cols-[1fr_auto_1fr] items-baseline gap-3">
                  <Side cell={row.offense} align="left" lead={row.edge === "offense"} />
                  <span className="text-center text-xs font-semibold text-text-sec">{row.label}<span className="block text-[10px] font-normal opacity-70">vs {row.defenseLabel}</span></span>
                  <Side cell={row.defense} align="right" lead={row.edge === "defense"} />
                </div>
                <div className="mt-2 flex h-1.5 gap-0.5 overflow-hidden rounded-full" role="img"
                  aria-label={`${row.label}: ${offenseTeam} ranks ${row.offense.rank} of ${row.offense.of}; ${defenseTeam} ${row.defenseLabel.toLowerCase()} ranks ${row.defense.rank} of ${row.defense.of}`}>
                  <span className="rounded-l-full bg-amber-200/80" style={{ flex: left }} />
                  <span className="rounded-r-full bg-sky-300/80" style={{ flex: right }} />
                </div>
              </li>
            );
          })}
        </ul>
        <p className="mt-1 text-[11px] text-text-sec">Rank 1 is best among the league&apos;s teams on both sides. Highlighted numbers mark a clear edge (8 or more places).</p>
      </div>
    </section>
  );
}
