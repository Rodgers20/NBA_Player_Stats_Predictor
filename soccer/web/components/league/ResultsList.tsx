"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { TeamLabel } from "@/components/TeamCrest";
import { cn, fmtDate } from "@/lib/format";
import type { Fixture, LeagueMatch, LiveMatch } from "@/lib/types";

const FILTERS = ["All", "Live", "Results", "Upcoming"] as const;
type Filter = (typeof FILTERS)[number];

/** The matches rail: filter chips over a date-grouped list. */
export function ResultsList({
  results, fixtures, live, allHref,
}: {
  results: LeagueMatch[];
  fixtures: Fixture[];
  live: LiveMatch[];
  allHref: string;
}) {
  const [filter, setFilter] = useState<Filter>("All");
  const liveKeys = new Set(live.map((l) => l.fixture_key));

  type Row = {
    key: string; date: string; time: string | null;
    home: string; away: string;
    score: string | null; state: "FT" | "LIVE" | "TIME";
    minutes?: number | null;
  };

  const resultRows: Row[] = results.slice(0, 14).map((m, i) => ({
    key: `r-${m.date}-${m.home_team}-${i}`, date: m.date, time: m.time,
    home: m.home_team, away: m.away_team,
    score: `${m.home_goals} – ${m.away_goals}`, state: "FT",
  }));

  const fixtureRows: Row[] = fixtures.slice(0, 14).map((f) => {
    const isLive = liveKeys.has(f.fixture_key) || f.live_status === "live";
    return {
      key: f.fixture_key, date: f.date, time: f.time,
      home: f.home_team, away: f.away_team, score: null,
      state: isLive ? "LIVE" : "TIME",
      minutes: live.find((l) => l.fixture_key === f.fixture_key)?.minutes_elapsed ?? null,
    };
  });

  const rows =
    filter === "Results" ? resultRows
    : filter === "Upcoming" ? fixtureRows.filter((r) => r.state === "TIME")
    : filter === "Live" ? fixtureRows.filter((r) => r.state === "LIVE")
    : [...fixtureRows, ...resultRows];

  const grouped = rows.reduce<Record<string, Row[]>>((acc, r) => {
    (acc[r.date] ??= []).push(r);
    return acc;
  }, {});

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-3 flex gap-1">
        {FILTERS.map((f) => {
          const count =
            f === "Live" ? fixtureRows.filter((r) => r.state === "LIVE").length : null;
          return (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={cn(
                "flex cursor-pointer items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium",
                "transition-colors duration-150",
                filter === f
                  ? "bg-flood text-white"
                  : "bg-elevated text-ink-secondary hover:text-ink",
              )}
            >
              {f}
              {count ? (
                <span className="rounded bg-live px-1 text-[9px] font-bold text-white tnum">
                  {count}
                </span>
              ) : null}
            </button>
          );
        })}
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1 max-h-[420px]">
        {Object.keys(grouped).length === 0 && (
          <p className="py-8 text-center text-[12px] text-ink-muted">
            Nothing to show for this filter.
          </p>
        )}
        {Object.entries(grouped).map(([date, list]) => (
          <div key={date}>
            <div className="mb-1.5 text-[10.5px] font-medium text-ink-muted">
              {fmtDate(date)}
            </div>
            <ul className="space-y-0.5">
              {list.map((r) => (
                <li
                  key={r.key}
                  className="grid grid-cols-[42px_1fr_auto_1fr] items-center gap-2 rounded-lg px-2 py-1.5 transition-colors duration-150 hover:bg-elevated/70"
                >
                  <span className="font-mono text-[10px] tnum text-ink-muted">
                    {r.state === "LIVE" ? (
                      <span className="flex items-center gap-1 font-semibold text-live">
                        <span className="live-dot !w-1.5 !h-1.5" aria-hidden />
                        {r.minutes != null ? `${r.minutes}'` : "LIVE"}
                      </span>
                    ) : r.state === "FT" ? "FT" : (r.time ?? "--:--")}
                  </span>
                  <TeamLabel name={r.home} size={16} align="right" className="justify-start text-[12px]" />
                  <span
                    className={cn(
                      "rounded px-1.5 py-0.5 font-mono text-[11px] tnum",
                      r.score ? "bg-elevated font-semibold" : "text-ink-muted",
                    )}
                  >
                    {r.score ?? "v"}
                  </span>
                  <TeamLabel name={r.away} size={16} className="text-[12px]" />
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <Link
        href={allHref}
        className="mt-3 flex cursor-pointer items-center justify-between rounded-lg bg-elevated/70 px-3 py-2 text-[12px] text-ink-secondary transition-colors duration-150 hover:text-ink"
      >
        View all matches <ArrowRight className="w-3.5 h-3.5" />
      </Link>
    </div>
  );
}
