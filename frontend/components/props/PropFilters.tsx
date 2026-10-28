"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { selectedStats, toggleStat } from "@/lib/prop-filters.mjs";
import { MultiPick } from "./MultiPick";

const MAX_PLAYERS = 10;

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
  window: string;
  ev: boolean;
  players: string;
  q: string;
}

/** Hit-rate windows a board can rank by; the keys match `hit_windows` on each prop. */
const HIT_WINDOWS = [["L5", "L5"], ["L10", "L10"], ["L20", "L20"], ["h2h", "H2H"], ["season", "Season"]] as const;

interface Props {
  filters: Filters;
  matchups: string[];
  statCounts?: Record<string, number>;
  onChange: (f: Partial<Filters>) => void;
  /** Sportsbooks that quote the loaded picks, and the ones the board is limited to (empty means any). */
  bookOptions: string[];
  books: string[];
  onBooksChange: (books: string[]) => void;
  /** Players on the loaded board, for the checklist. */
  playerOptions: string[];
  /** Headshot per player name, shown in the checklist. */
  playerImages?: Record<string, string | null | undefined>;
}

export function PropFilters({ filters, matchups, statCounts, onChange, bookOptions, books, onBooksChange, playerOptions, playerImages }: Props) {
  const isOn = (value: string) => (value === "" ? filters.stat === "" : selectedStats(filters.stat).includes(value));
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
          maxLength={40} className="ql-control min-h-9 w-full px-3 py-2 text-sm text-text-pri placeholder:text-text-sec" />
      </label>
      <p className="ql-data-label mb-2 text-text-sec">STATISTIC</p>
      <div className="mb-3 flex flex-wrap gap-1.5" role="group" aria-label="Filter by stat">
        {STATS.map(([value, label]) => (
          <button
            key={label}
            type="button"
            aria-pressed={isOn(value)}
            onClick={() => onChange({ stat: toggleStat(filters.stat, value) })}
            className={cn(
              "ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400",
              isOn(value)
                ? "ql-chip-active"
                : "text-text-sec hover:border-teal-400/45 hover:text-ink"
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {statCounts && Object.keys(statCounts).length > 0 && (
        <div className="mb-3 flex flex-wrap gap-1.5" aria-label="Available props by stat">
          {STATS.filter(([value]) => value && statCounts[value]).map(([value, label]) => (
            <span key={value} className="ql-data-label rounded border border-ink/10 px-2.5 py-0.5 text-[10px] text-text-sec">
              {label} {statCounts[value]}
            </span>
          ))}
        </div>
      )}

      {isLineMarket(filters.stat) && <p className="border-t border-ink/10 pt-3 text-xs text-text-sec">Whole-game lines from the game model. Side, location and star filters apply to player props only.</p>}
      {!isLineMarket(filters.stat) && <div className="flex flex-wrap items-end gap-x-5 gap-y-3 border-t border-ink/10 pt-3">
        <div role="group" aria-label="Filter by direction">
          <p className="ql-data-label mb-1.5 text-text-sec">SIDE</p>
          <div className="flex items-center gap-1.5">
          {(["All", "Over", "Under"] as const).map((direction) => (
            <button
              key={direction}
              type="button"
              aria-pressed={filters.direction === direction}
              onClick={() => onChange({ direction })}
              className={cn(
                "ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400",
                filters.direction === direction
                  ? "ql-chip-active"
                  : "text-text-sec hover:border-teal-400/45 hover:text-ink"
              )}
            >
              {direction === "Over" ? "Overs ↑" : direction === "Under" ? "Unders ↓" : "All"}
            </button>
          ))}
          </div>
        </div>

        <div role="group" aria-label="Filter by location">
          <p className="ql-data-label mb-1.5 text-text-sec">LOCATION</p>
          <div className="flex items-center gap-1.5">
          {(["All", "Home", "Away"] as const).map((location) => (
            <button
              key={location}
              type="button"
              aria-pressed={filters.location === location}
              onClick={() => onChange({ location })}
              className={cn(
                "ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400",
                filters.location === location
                  ? "ql-chip-active"
                  : "text-text-sec hover:border-teal-400/45 hover:text-ink"
              )}
            >
              {location}
            </button>
          ))}
          </div>
        </div>

        <div role="group" aria-label="Hit rate window">
          <p className="ql-data-label mb-1.5 text-text-sec">HIT WINDOW</p>
          <div className="flex items-center gap-1.5">
          {HIT_WINDOWS.map(([value, label]) => (
            <button
              key={value}
              type="button"
              aria-pressed={filters.window === value}
              onClick={() => onChange({ window: value })}
              title={`Rank and show hit rate over ${label === "H2H" ? "head-to-head games" : label === "Season" ? "this season" : `the last ${label.slice(1)} games`}`}
              className={cn(
                "ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400",
                filters.window === value
                  ? "ql-chip-active"
                  : "text-text-sec hover:border-teal-400/45 hover:text-ink"
              )}
            >
              {label}
            </button>
          ))}
          </div>
        </div>

        <div role="group" aria-label="More filters">
          <p className="ql-data-label mb-1.5 text-text-sec">NARROW</p>
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              aria-pressed={filters.ev}
              onClick={() => onChange({ ev: !filters.ev })}
              title="Only picks with a positive expected value at the shown price"
              className={cn(
                "ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400",
                filters.ev ? "ql-chip-active" : "text-text-sec hover:border-teal-400/45 hover:text-ink"
              )}
            >
              Only +EV
            </button>
            <MultiPick label="Players" options={playerOptions} selected={filters.players ? filters.players.split("|") : []} searchable max={MAX_PLAYERS} images={playerImages}
              onChange={(next) => onChange({ players: next.join("|") })} emptyText="Players appear once the board loads." />
            <MultiPick label="Sportsbooks" options={bookOptions} selected={books} onChange={onBooksChange}
              emptyText="No sportsbook quotes on the loaded picks." />
          </div>
        </div>

        <div className="ml-auto flex flex-wrap items-end gap-3">
          <label className="ql-data-label flex flex-col gap-1.5 text-text-sec">
            GAME
            <select
              value={filters.game}
              onChange={(event) => onChange({ game: event.target.value })}
              className="ql-control min-h-8 min-w-36 px-2.5 py-1.5 text-xs font-medium normal-case tracking-normal text-text-pri"
            >
              <option value="">All Games</option>
              {matchups.map((matchup) => <option key={matchup} value={matchup}>{matchup}</option>)}
            </select>
          </label>
          <label className="ql-data-label flex flex-col gap-1.5 text-text-sec">
            SORT
            <select
              value={filters.sort}
              onChange={(event) => onChange({ sort: event.target.value })}
              className="ql-control min-h-8 min-w-40 px-2.5 py-1.5 text-xs font-medium normal-case tracking-normal text-text-pri"
            >
              <option value="hit_rate">Highest Hit Rate</option>
              <option value="ev">Highest EV</option>
              <option value="edge">Biggest Edge</option>
            </select>
          </label>
          <label className="ql-data-label flex flex-col gap-1.5 text-text-sec">
            MIN STARS
            <select
              value={filters.minStars}
              onChange={(event) => onChange({ minStars: Number(event.target.value) })}
              className="ql-control min-h-8 min-w-24 px-2.5 py-1.5 text-xs font-medium normal-case tracking-normal text-text-pri"
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
