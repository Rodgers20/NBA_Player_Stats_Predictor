"use client";

import { useId, useState } from "react";
import { Home, SlidersHorizontal } from "lucide-react";
import { RangeSlider } from "@/components/ui/RangeSlider";
import {
  DEFAULT_GAME_FILTERS, MARGIN_CAP, activeFilterCount, histogram, minutesRange, teammateOptions,
  type GameFilters, type RestFilter,
} from "@/lib/game-filters.mjs";
import type { ChartRecord } from "@/lib/types";

const REST: [RestFilter, string][] = [["any", "Any rest"], ["0", "Back to back"], ["1", "1 day"], ["2", "2 days"], ["3+", "3+ days"]];
const select = "ql-control min-h-9 w-full px-3 py-2 text-sm text-text-pri focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]";
const label = "ql-data-label mb-1.5 block";

interface Props {
  /** Every game for this player and stat, before any filter: the histograms and teammate list describe the whole sample. */
  games: ChartRecord[];
  value: GameFilters;
  onChange: (next: GameFilters) => void;
  /** Games left after the filters, for the summary line. */
  matching: number;
}

/** Narrow the games the chart and hit rates are computed from: venue, rest, teammates, margin, result and minutes. */
export function GameFilterPanel({ games, value, onChange, matching }: Props) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const active = activeFilterCount(value);
  const teammates = teammateOptions(games);
  const [minLow, minHigh] = minutesRange(games);
  const minutes = value.minutes ?? [minLow, minHigh];
  const set = (patch: Partial<GameFilters>) => onChange({ ...value, ...patch });

  return (
    <section className="ql-panel p-4 sm:p-5" aria-label="Game filters">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <button type="button" aria-expanded={open} aria-controls={panelId} onClick={() => setOpen(current => !current)}
          className="ql-control inline-flex min-h-9 items-center gap-2 px-3 py-2 text-sm font-semibold text-text-pri focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">
          <SlidersHorizontal size={16} aria-hidden="true" /> Game filters
          {active > 0 && <span className="rounded-full bg-[#59e0c8] px-1.5 text-[11px] font-bold text-[#0d151b]" aria-label={`${active} active`}>{active}</span>}
        </button>
        <p className="text-xs text-text-sec" role="status">{active > 0 ? `${matching} of ${games.length} games match` : `${games.length} games`}</p>
      </div>

      {open && (
        <div id={panelId} className="mt-4 grid gap-5 md:grid-cols-2">
          <div>
            <span className={label}>VENUE</span>
            <button type="button" role="switch" aria-checked={value.homeOnly} onClick={() => set({ homeOnly: !value.homeOnly })}
              className="ql-control inline-flex min-h-9 w-full items-center justify-between gap-2 px-3 py-2 text-sm text-text-pri focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">
              <span className="inline-flex items-center gap-2"><Home size={16} aria-hidden="true" /> Show only home games</span>
              <span className={`h-5 w-9 rounded-full p-0.5 transition-colors ${value.homeOnly ? "bg-[#59e0c8]" : "bg-white/15"}`} aria-hidden="true">
                <span className={`block h-4 w-4 rounded-full bg-[#eaf8f6] transition-transform ${value.homeOnly ? "translate-x-4" : ""}`} />
              </span>
            </button>
          </div>
          <label>
            <span className={label}>DAYS OF REST</span>
            <select className={select} value={value.rest} onChange={event => set({ rest: event.target.value as RestFilter })}>
              {REST.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
            </select>
          </label>

          <label>
            <span className={label}>GAMES WITH PLAYER</span>
            <select className={select} value={value.withPlayer} disabled={!teammates.length} onChange={event => set({ withPlayer: event.target.value })}>
              <option value="">Any teammate</option>
              {teammates.map(name => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <label>
            <span className={label}>GAMES WITHOUT PLAYER</span>
            <select className={select} value={value.withoutPlayer} disabled={!teammates.length} onChange={event => set({ withoutPlayer: event.target.value })}>
              <option value="">Any teammate</option>
              {teammates.map(name => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          {!teammates.length && <p className="text-xs text-text-sec md:col-span-2">No rotation teammate missed a game in this sample, so there is nothing to split on.</p>}

          <div>
            <div className="mb-1.5 flex items-center justify-between gap-3">
              <span className="ql-data-label">MARGIN OF RESULT</span>
              <label className="flex items-center gap-2 text-xs text-text-sec">
                <span className="sr-only">Win or loss</span>
                <select className="ql-control min-h-8 px-2 py-1 text-xs text-text-pri" value={value.result} onChange={event => set({ result: event.target.value as GameFilters["result"] })}>
                  <option value="any">Any W/L</option><option value="W">Wins</option><option value="L">Losses</option>
                </select>
              </label>
            </div>
            <RangeSlider label="Margin" min={0} max={MARGIN_CAP} low={value.margin[0]} high={value.margin[1]}
              bars={histogram(games.map(game => (game.margin == null ? null : Math.abs(game.margin))), 0, MARGIN_CAP, MARGIN_CAP)}
              format={(point, edge) => (edge === "high" && point >= MARGIN_CAP ? `${point}+` : String(point))}
              onChange={margin => set({ margin })} />
          </div>
          <div>
            <span className={label}>MINUTES PER GAME</span>
            <RangeSlider label="Minutes" min={minLow} max={minHigh} low={minutes[0]} high={minutes[1]}
              bars={histogram(games.map(game => game.minutes), minLow, minHigh, 14)}
              onChange={([low, high]) => set({ minutes: low <= minLow && high >= minHigh ? null : [low, high] })} />
          </div>

          <div className="md:col-span-2 md:text-center">
            <button type="button" disabled={active === 0} onClick={() => onChange(DEFAULT_GAME_FILTERS)}
              className="text-xs font-bold tracking-[.14em] text-text-sec enabled:hover:text-text-pri disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">RESET ALL</button>
          </div>
        </div>
      )}
    </section>
  );
}
