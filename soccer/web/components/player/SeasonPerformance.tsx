"use client";

import { useState } from "react";
import type { PlayerSpell, RadarMetric } from "@/lib/types";

/**
 * The ranked stat sheet, with FotMob's Total / Per 90 toggle.
 *
 * Bar length is a *percentile* against positional peers, not the raw value —
 * that is what makes "18 shots" legible without knowing what a normal shot
 * count is. Colour carries the same information in a second channel so the
 * ranking survives greyscale and the common forms of colour blindness.
 *
 * Rows whose statistic has no peer ranking still draw a bar, scaled within the
 * section, so the sheet reads as one object rather than a ragged mix of bars
 * and "no rank" text. Those bars are drawn in a muted tone and labelled on
 * hover so they are not mistaken for a percentile.
 */
type Row = {
  label: string;
  total: number | null;
  per90?: number | null;
  percentile?: number | null;
  suffix?: string;
  /** A rate is already normalised; per-90 must not divide it again. */
  isRate?: boolean;
};

export function SeasonPerformance({
  spell, accent,
}: { spell: PlayerSpell; accent: string }) {
  const [mode, setMode] = useState<"total" | "per90">("total");
  const s = spell.stats;
  const p = spell.per90 ?? {};
  const rank = (key: string) =>
    spell.radar?.metrics.find((m: RadarMetric) => m.key === key)?.percentile ?? null;

  const sections: { title: string; rows: Row[] }[] = [
    {
      title: "Shooting",
      rows: [
        { label: "Goals", total: s.goals, per90: p.goals, percentile: rank("goals") },
        { label: "Expected goals (xG)", total: s.xg, per90: p.xg },
        { label: "Shots", total: s.shots, per90: p.shots, percentile: rank("shots") },
        { label: "Shots on target", total: s.shots_on_target },
        { label: "Shot accuracy", total: s.shot_accuracy, suffix: "%", isRate: true },
      ],
    },
    {
      title: "Passing",
      rows: [
        { label: "Assists", total: s.assists, per90: p.assists },
        { label: "Chances created", total: s.chances_created, per90: p.chances_created,
          percentile: rank("chances_created") },
        { label: "Passes completed", total: s.passes_completed },
        { label: "Pass accuracy", total: s.pass_accuracy, suffix: "%", isRate: true },
      ],
    },
    {
      title: "Possession",
      rows: [
        { label: "Touches", total: s.touches, per90: p.touches, percentile: rank("touches") },
        { label: "Touches in final third", total: s.touches_final_third },
        { label: "Successful dribbles", total: s.dribbles_completed, per90: p.dribbles_completed },
        { label: "Dribble success", total: s.dribble_success, suffix: "%", isRate: true },
        { label: "Dispossessed", total: s.dispossessed },
        { label: "Miscontrols", total: s.miscontrols },
      ],
    },
    {
      title: "Defending",
      rows: [
        { label: "Duels won", total: s.duels_won, per90: p.duels_won, percentile: rank("duels_won") },
        { label: "Duel success", total: s.duel_success, suffix: "%", isRate: true },
        { label: "Interceptions", total: s.interceptions, per90: p.interceptions,
          percentile: rank("interceptions") },
        { label: "Ball recoveries", total: s.recoveries, per90: p.recoveries },
        { label: "Clearances", total: s.clearances },
        { label: "Blocks", total: s.blocks },
        { label: "Fouls won", total: s.fouls_won },
        { label: "Fouls committed", total: s.fouls_committed },
      ],
    },
  ];

  return (
    <div className="p-3 sm:p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[12px] text-ink-muted">
          Minutes played: {s.minutes_estimated?.toLocaleString() ?? "–"}
          <span className="ml-1 opacity-70">(estimated)</span>
        </p>
        <div role="group" aria-label="Statistic scale"
             className="flex rounded-full border border-stroke bg-elevated p-0.5">
          {(["total", "per90"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
              className={`rounded-full px-3.5 py-1 text-[12px] font-semibold transition-colors ${
                mode === m ? "bg-ink text-base" : "text-ink-muted hover:text-ink"
              }`}
            >
              {m === "total" ? "Total" : "Per 90"}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        {sections.map((section, i) => {
          const values = section.rows
            .map((r) => valueFor(r, mode))
            .filter((v): v is number => v !== null && v !== undefined);
          const localMax = Math.max(...values, 1);

          return (
            <section key={section.title} className="border-t border-stroke pt-3 first:border-t-0 first:pt-0">
              <div className="mb-1 flex items-baseline justify-between">
                <h3 className="font-display text-[13px] font-semibold text-ink">{section.title}</h3>
                {i === 0 && <span className="text-[11px] text-ink-muted">Rank vs peers</span>}
              </div>
              {section.rows.map((r) => (
                <StatRow key={r.label} row={r} mode={mode} localMax={localMax}
                         accent={accent} />
              ))}
            </section>
          );
        })}
      </div>
    </div>
  );
}

function valueFor(row: Row, mode: "total" | "per90"): number | null {
  if (mode === "total" || row.isRate) return row.total;
  return row.per90 ?? null;
}

function StatRow({
  row, mode, localMax,
}: { row: Row; mode: "total" | "per90"; localMax: number; accent: string }) {
  const value = valueFor(row, mode);
  const unavailable = mode === "per90" && !row.isRate && row.per90 === undefined;
  const shown = unavailable ? "–" : value === null || value === undefined ? "–" : `${value}${row.suffix ?? ""}`;

  const ranked = row.percentile !== null && row.percentile !== undefined;
  const width = ranked
    ? Math.max(3, Math.min(100, row.percentile!))
    : value === null || value === undefined
      ? 0
      : Math.max(3, Math.min(100, (value / localMax) * 100));

  return (
    <div className="grid grid-cols-[1fr_auto_minmax(90px,190px)] items-center gap-3 py-[7px]">
      <span className="truncate text-[13px] text-ink-secondary">{row.label}</span>
      <span className="tabular-nums text-[13px] font-semibold text-ink">{shown}</span>
      <span className="flex items-center gap-2">
        <span aria-hidden className="h-4 w-px shrink-0 bg-stroke" />
        <span className="relative h-[7px] flex-1 overflow-hidden rounded-full bg-elevated"
              title={ranked
                ? `${row.percentile}th percentile among positional peers`
                : "Scaled within this section — no peer ranking for this statistic"}>
          <span
            className="absolute inset-y-0 left-0 rounded-full"
            style={{
              width: `${width}%`,
              background: ranked ? bandColor(row.percentile!) : "var(--color-stroke-hi)",
            }}
          />
        </span>
      </span>
    </div>
  );
}

function bandColor(percentile: number): string {
  if (percentile >= 70) return "var(--color-form-w, #09AF51)";
  if (percentile >= 40) return "var(--color-series-4, #C0851F)";
  return "var(--color-series-2, #FC4540)";
}
