"use client";

import type { StatRow } from "@/lib/types";

/**
 * Mirrored bars growing outward from a centred label — the "MATCH STATS" panel
 * from both reference designs.
 *
 * Mark spec: thin bars, 4px rounded data-ends anchored at the centre, and a 2px
 * surface gap between the two fills so they never touch. Values sit in text ink,
 * never the series colour; the coloured bar beside them carries identity.
 */
export function StatBars({ rows, homeTeam, awayTeam }: {
  rows: StatRow[]; homeTeam: string; awayTeam: string;
}) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between text-[11px] text-ink-secondary">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full" style={{ background: "var(--color-home)" }} />
          {homeTeam}
        </span>
        <span className="flex items-center gap-1.5">
          {awayTeam}
          <span className="w-2 h-2 rounded-full" style={{ background: "var(--color-away)" }} />
        </span>
      </div>

      {rows.map((r) => {
        const total = r.home + r.away;
        const hp = total > 0 ? (r.home / total) * 100 : 50;
        const ap = total > 0 ? (r.away / total) * 100 : 50;
        return (
          <div key={r.label} className="group">
            <div className="flex items-center gap-2 text-[11px]">
              <span className="w-11 shrink-0 text-right font-mono tnum text-ink">
                {fmt(r.home, r.unit)}
              </span>
              <span className="flex-1 text-center text-ink-secondary">{r.label}</span>
              <span className="w-11 shrink-0 font-mono tnum text-ink">
                {fmt(r.away, r.unit)}
              </span>
            </div>
            <div className="mt-1 flex items-center gap-[2px]">
              <div className="flex flex-1 justify-end h-[5px] rounded-l-full bg-[var(--color-stroke)]/60 overflow-hidden">
                <div
                  className="h-full rounded-l-full transition-[width] duration-500"
                  style={{ width: `${hp}%`, background: "var(--color-home)" }}
                />
              </div>
              <div className="flex flex-1 h-[5px] rounded-r-full bg-[var(--color-stroke)]/60 overflow-hidden">
                <div
                  className="h-full rounded-r-full transition-[width] duration-500"
                  style={{ width: `${ap}%`, background: "var(--color-away)" }}
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

const fmt = (v: number, unit: string) =>
  unit === "%" ? `${v.toFixed(0)}%` : Number.isInteger(v) ? `${v}` : v.toFixed(2);
