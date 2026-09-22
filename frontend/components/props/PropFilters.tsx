"use client";

import { cn } from "@/lib/utils";

const STATS = ["All", "PTS", "AST", "REB", "FG3M", "STL", "BLK", "COMBO"];
const DIRECTIONS = ["Over", "Under", "All"];
const SORTS = [
  { value: "ev",       label: "Best EV" },
  { value: "hit_rate", label: "Hit Rate" },
];

interface Filters {
  stat:       string;
  direction:  string;
  sort:       string;
  game:       string;
  locksOnly:  boolean;
}

interface Props {
  filters:    Filters;
  matchups:   string[];
  onChange:   (f: Partial<Filters>) => void;
}

export function PropFilters({ filters, matchups, onChange }: Props) {
  return (
    <div className="flex flex-col gap-3 pb-1">
      {/* Stat chips */}
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
        {STATS.map((s) => {
          const active = filters.stat === (s === "All" ? "" : s);
          return (
            <button
              key={s}
              onClick={() => onChange({ stat: s === "All" ? "" : s })}
              className={cn(
                "flex-shrink-0 text-xs font-semibold px-3 py-1.5 rounded-full border transition-colors",
                active
                  ? "bg-[#2dd4bf]/15 border-[#2dd4bf]/50 text-[#2dd4bf]"
                  : "bg-transparent border-white/10 text-[#8ca0c0]"
              )}
            >
              {s}
            </button>
          );
        })}
      </div>

      {/* Second row: direction, sort, game, locks */}
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide items-center">
        {DIRECTIONS.map((d) => {
          const active = filters.direction === d;
          return (
            <button
              key={d}
              onClick={() => onChange({ direction: d })}
              className={cn(
                "flex-shrink-0 text-xs font-medium px-3 py-1.5 rounded-full border transition-colors",
                active
                  ? "bg-[#f97066]/15 border-[#f97066]/40 text-[#f97066]"
                  : "bg-transparent border-white/10 text-[#8ca0c0]"
              )}
            >
              {d}
            </button>
          );
        })}

        <div className="w-px h-4 bg-white/10 flex-shrink-0" />

        {SORTS.map((s) => (
          <button
            key={s.value}
            onClick={() => onChange({ sort: s.value })}
            className={cn(
              "flex-shrink-0 text-xs font-medium px-3 py-1.5 rounded-full border transition-colors",
              filters.sort === s.value
                ? "bg-[#a78bfa]/15 border-[#a78bfa]/40 text-[#a78bfa]"
                : "bg-transparent border-white/10 text-[#8ca0c0]"
            )}
          >
            {s.label}
          </button>
        ))}

        <div className="w-px h-4 bg-white/10 flex-shrink-0" />

        <button
          onClick={() => onChange({ locksOnly: !filters.locksOnly })}
          className={cn(
            "flex-shrink-0 text-xs font-medium px-3 py-1.5 rounded-full border transition-colors",
            filters.locksOnly
              ? "bg-[#f59e0b]/15 border-[#f59e0b]/40 text-[#f59e0b]"
              : "bg-transparent border-white/10 text-[#8ca0c0]"
          )}
        >
          🔒 Locks
        </button>

        {matchups.length > 0 && (
          <select
            value={filters.game}
            onChange={(e) => onChange({ game: e.target.value })}
            className="flex-shrink-0 text-xs font-medium px-3 py-1.5 rounded-full border border-white/10 bg-transparent text-[#8ca0c0] appearance-none cursor-pointer"
          >
            <option value="">All Games</option>
            {matchups.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}
