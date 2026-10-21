"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

const STATS = [
  ["", "All"], ["PTS", "Points"], ["AST", "Assists"], ["REB", "Rebounds"],
  ["FG3M", "3-Ptrs"], ["BLK", "Blocks"], ["STL", "Steals"],
  ["PTS+AST", "Pts+Ast"], ["PTS+REB", "Pts+Reb"],
  ["REB+AST", "Reb+Ast"], ["PTS+REB+AST", "PRA"],
  ["DD", "Dbl-Dbl"], ["TD", "Tri-Dbl"],
  ["ML", "Moneyline"], ["SPREAD", "Spread"], ["TOTAL", "Totals"],
] as const;

/** Stat filters that switch the board from player props to whole-game lines. */
export const LINE_MARKETS = ["ML", "SPREAD", "TOTAL"] as const;
export type LineMarket = (typeof LINE_MARKETS)[number];
export const isLineMarket = (stat: string): stat is LineMarket => (LINE_MARKETS as readonly string[]).includes(stat);

export interface Filters {
  stat: string;
  direction: string;
  location: string;
  sort: string;
  game: string;
  locksOnly: boolean;
  minStars: number;
  q: string;
}

interface Props {
  filters: Filters;
  matchups: string[];
  statCounts?: Record<string, number>;
  onChange: (f: Partial<Filters>) => void;
}

export function PropFilters({ filters, matchups, statCounts, onChange }: Props) {
  const [draft, setDraft] = useState(filters.q);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const latest = useRef(onChange);
  // Always call the newest handler (it merges with the current URL), and never fire after unmount.
  useEffect(() => { latest.current = onChange; });
  useEffect(() => () => clearTimeout(timer.current), []);
  const typeSearch = (value: string) => {
    setDraft(value);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => latest.current({ q: value }), 250); // keep the URL calm while typing
  };
  return (
    <div className="ql-panel mb-5 p-3 sm:p-4">
      <label className="mb-3 block">
        <span className="sr-only">Search players or teams</span>
        <input type="search" value={draft} onChange={(event) => typeSearch(event.target.value)} placeholder="Search player or team"
          maxLength={40} className="ql-control min-h-9 w-full px-3 py-2 text-sm text-[#eaf8f6] placeholder:text-[#a9bec0]" />
      </label>
      <p className="ql-data-label mb-2 text-[#a9bec0]">STATISTIC</p>
      <div className="mb-3 flex flex-wrap gap-1.5" role="group" aria-label="Filter by stat">
        {STATS.map(([value, label]) => (
          <button
            key={label}
            type="button"
            aria-pressed={filters.stat === value}
            onClick={() => onChange({ stat: value })}
            className={cn(
              "ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]",
              filters.stat === value
                ? "ql-chip-active"
                : "text-[#a9bec0] hover:border-[#59e0c8]/45 hover:text-white"
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {statCounts && Object.keys(statCounts).length > 0 && (
        <div className="mb-3 flex flex-wrap gap-1.5" aria-label="Available props by stat">
          {STATS.filter(([value]) => value && statCounts[value]).map(([value, label]) => (
            <span key={value} className="ql-data-label rounded border border-white/10 px-2.5 py-0.5 text-[10px] text-[#a9bec0]">
              {label} {statCounts[value]}
            </span>
          ))}
        </div>
      )}

      {isLineMarket(filters.stat) && <p className="border-t border-white/10 pt-3 text-xs text-[#a9bec0]">Whole-game lines from the game model. Side, location and star filters apply to player props only.</p>}
      {!isLineMarket(filters.stat) && <div className="flex flex-wrap items-end gap-x-5 gap-y-3 border-t border-white/10 pt-3">
        <div role="group" aria-label="Filter by direction">
          <p className="ql-data-label mb-1.5 text-[#a9bec0]">SIDE</p>
          <div className="flex items-center gap-1.5">
          {(["All", "Over", "Under"] as const).map((direction) => (
            <button
              key={direction}
              type="button"
              aria-pressed={filters.direction === direction}
              onClick={() => onChange({ direction })}
              className={cn(
                "ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]",
                filters.direction === direction
                  ? "ql-chip-active"
                  : "text-[#a9bec0] hover:border-[#59e0c8]/45 hover:text-white"
              )}
            >
              {direction === "Over" ? "Overs ↑" : direction === "Under" ? "Unders ↓" : "All"}
            </button>
          ))}
          </div>
        </div>

        <div role="group" aria-label="Filter by location">
          <p className="ql-data-label mb-1.5 text-[#a9bec0]">LOCATION</p>
          <div className="flex items-center gap-1.5">
          {(["All", "Home", "Away"] as const).map((location) => (
            <button
              key={location}
              type="button"
              aria-pressed={filters.location === location}
              onClick={() => onChange({ location })}
              className={cn(
                "ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]",
                filters.location === location
                  ? "ql-chip-active"
                  : "text-[#a9bec0] hover:border-[#59e0c8]/45 hover:text-white"
              )}
            >
              {location}
            </button>
          ))}
          </div>
        </div>

        <div className="ml-auto flex flex-wrap items-end gap-3">
          <label className="ql-data-label flex flex-col gap-1.5 text-[#a9bec0]">
            GAME
            <select
              value={filters.game}
              onChange={(event) => onChange({ game: event.target.value })}
              className="ql-control min-h-8 min-w-36 px-2.5 py-1.5 text-xs font-medium normal-case tracking-normal text-[#eaf8f6]"
            >
              <option value="">All Games</option>
              {matchups.map((matchup) => <option key={matchup} value={matchup}>{matchup}</option>)}
            </select>
          </label>
          <label className="ql-data-label flex flex-col gap-1.5 text-[#a9bec0]">
            SORT
            <select
              value={filters.sort}
              onChange={(event) => onChange({ sort: event.target.value })}
              className="ql-control min-h-8 min-w-40 px-2.5 py-1.5 text-xs font-medium normal-case tracking-normal text-[#eaf8f6]"
            >
              <option value="hit_rate">Highest Hit Rate</option>
              <option value="ev">Highest EV</option>
              <option value="edge">Biggest Edge</option>
            </select>
          </label>
          <label className="ql-data-label flex flex-col gap-1.5 text-[#a9bec0]">
            MIN STARS
            <select
              value={filters.minStars}
              onChange={(event) => onChange({ minStars: Number(event.target.value) })}
              className="ql-control min-h-8 min-w-24 px-2.5 py-1.5 text-xs font-medium normal-case tracking-normal text-[#eaf8f6]"
            >
              <option value={0}>Any</option>
              {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}+ ★</option>)}
            </select>
          </label>
        </div>
      </div>}
    </div>
  );
}
