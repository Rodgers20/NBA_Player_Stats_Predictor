"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowRight } from "lucide-react";
import { cn, fmtDate } from "@/lib/format";
import type { Match } from "@/lib/types";

const FILTERS = ["All", "Results", "Upcoming"] as const;
type Filter = (typeof FILTERS)[number];

export function MatchList({ matches }: { matches: Match[] }) {
  const [filter, setFilter] = useState<Filter>("All");
  const shown = matches.slice(0, 10);

  const grouped = shown.reduce<Record<string, Match[]>>((acc, m) => {
    (acc[m.match_date] ??= []).push(m);
    return acc;
  }, {});

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="mb-3 flex gap-1">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={cn(
              "rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors",
              filter === f
                ? "bg-flood text-white"
                : "bg-elevated text-ink-secondary hover:text-ink",
            )}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto space-y-3 pr-1">
        {Object.entries(grouped).map(([date, list]) => (
          <div key={date}>
            <div className="mb-1.5 text-[10.5px] font-medium text-ink-muted">{fmtDate(date)}</div>
            <ul className="space-y-0.5">
              {list.map((m) => (
                <li key={m.match_id}>
                  <Link
                    href={`/match/${m.match_id}`}
                    className="grid grid-cols-[26px_1fr_auto_1fr] items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-elevated/80"
                  >
                    <span className="text-[9.5px] font-semibold text-ink-muted">FT</span>
                    <span className="truncate text-right text-[12px]">{m.home_team}</span>
                    <span className="rounded bg-elevated px-1.5 py-0.5 font-mono text-[11px] font-semibold tnum">
                      {m.home_score} – {m.away_score}
                    </span>
                    <span className="truncate text-[12px]">{m.away_team}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <Link
        href="/matches"
        className="mt-3 flex items-center justify-between rounded-lg bg-elevated/70 px-3 py-2 text-[12px] text-ink-secondary hover:text-ink"
      >
        View all matches <ArrowRight className="w-3.5 h-3.5" />
      </Link>
    </div>
  );
}
