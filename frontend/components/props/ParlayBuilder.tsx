"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSlip } from "@/store/slip";
import { formatOdds, summarizeSlip } from "@/lib/bet-slip.mjs";
import { buildParlay, canBuildFrom, propLeg, type ParlayRisk } from "@/lib/parlay-builder.mjs";
import type { Prop } from "@/lib/types";
import { MultiPick } from "./MultiPick";

const LEG_COUNTS = [2, 3, 4, 5, 6];
const RISKS: { value: ParlayRisk; label: string; hint: string }[] = [
  { value: "safe", label: "Safest", hint: "Highest model chance per leg" },
  { value: "edge", label: "Best edge", hint: "Biggest gap between the model and the book" },
  { value: "payout", label: "Best value", hint: "Highest expected return per dollar" },
];

interface Props {
  props: Prop[];
  league: "nba" | "wnba";
  date: string;
}

/** Builds a parlay from the model's priced picks with plain rules, then hands it to the slip, which prices correlation. */
export function ParlayBuilder({ props, league, date }: Props) {
  const [open, setOpen] = useState(false);
  const [legCount, setLegCount] = useState(3);
  const [risk, setRisk] = useState<ParlayRisk>("safe");
  const [include, setInclude] = useState<string[]>([]);
  const { legs: slipLegs, stake, load } = useSlip();

  const players = [...new Set(props.filter(canBuildFrom).map(prop => prop.player))].sort();
  if (players.length < 2) return null;
  const picks = buildParlay(props, { legs: legCount, risk, include });
  const legs = picks.map(prop => propLeg(prop, league, date));
  const summary = summarizeSlip(legs);
  const chip = (active: boolean) => cn("ql-chip min-h-8 px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]",
    active ? "ql-chip-active" : "text-[#a9bec0] hover:border-[#59e0c8]/45 hover:text-white");

  return (
    <section className="ql-panel mb-5" aria-label="Parlay builder">
      <button type="button" aria-expanded={open} onClick={() => setOpen(value => !value)}
        className="flex w-full items-center justify-between px-5 py-4 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">
        <span>
          <span className="block text-sm font-bold text-[#eaf8f6]">Build me a parlay</span>
          <span className="block text-xs text-[#a9bec0]">Picks legs from the model&apos;s priced picks: one per player, spread across games.</span>
        </span>
        <ChevronDown size={16} aria-hidden="true" className={cn("shrink-0 text-[#59e0c8] transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="border-t border-white/10 px-5 py-4">
          <div className="flex flex-wrap items-end gap-5">
            <div role="group" aria-label="Number of legs">
              <p className="ql-data-label mb-1.5 text-[#a9bec0]">LEGS</p>
              <div className="flex gap-1.5">
                {LEG_COUNTS.map(count => <button key={count} type="button" aria-pressed={legCount === count} onClick={() => { setLegCount(count); setInclude(include.slice(0, count)); }} className={chip(legCount === count)}>{count}</button>)}
              </div>
            </div>
            <div role="group" aria-label="Risk">
              <p className="ql-data-label mb-1.5 text-[#a9bec0]">BUILD FOR</p>
              <div className="flex flex-wrap gap-1.5">
                {RISKS.map(item => <button key={item.value} type="button" title={item.hint} aria-pressed={risk === item.value} onClick={() => setRisk(item.value)} className={chip(risk === item.value)}>{item.label}</button>)}
              </div>
            </div>
            <MultiPick label="Must include" options={players} selected={include} onChange={setInclude} searchable max={legCount} />
          </div>

          <ol className="mt-4 divide-y divide-white/[0.06] rounded-lg border border-white/10">
            {picks.map((prop, index) => (
              <li key={`${prop.player}-${prop.stat}`} className="grid grid-cols-[20px_minmax(0,1fr)_auto_auto] items-center gap-3 px-3 py-2.5 text-sm">
                <span className="text-xs font-bold text-[#a9bec0]">{index + 1}</span>
                <span className="min-w-0">
                  <span className="block truncate font-semibold text-[#eaf8f6]">{prop.player}</span>
                  <span className="block truncate text-xs text-[#a9bec0]"><strong className="text-[#59e0c8]">{prop.direction} {prop.line} {prop.stat_label || prop.stat}</strong> · {prop.game_matchup}</span>
                </span>
                <span className="text-xs font-bold text-[#59e0c8]">{Math.round((prop.model_prob ?? 0) * 100)}%</span>
                <span className="w-12 text-right text-xs font-semibold text-[#eaf8f6]">{formatOdds(prop.price)}</span>
              </li>
            ))}
          </ol>
          {picks.length < legCount && <p className="mt-2 text-xs text-[#f5ba64]">Only {picks.length} legs fit these rules on this board.</p>}

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-[#a9bec0]">
              {picks.length}-leg parlay <strong className="text-[#eaf8f6]">{formatOdds(summary.american)}</strong>
              {summary.probability !== null && <> · model chance <strong className="text-[#eaf8f6]">{(summary.probability * 100).toFixed(1)}%</strong> if the legs were independent</>}
              {summary.correlated && <> · shares a game, so the slip adjusts for correlation</>}
            </p>
            <button type="button" disabled={picks.length < 2} onClick={() => load(legs, "parlay", stake)}
              className="ql-control px-4 py-2 text-xs font-bold text-[#59e0c8] hover:bg-[#59e0c8]/10 disabled:opacity-50">
              {slipLegs.length ? `Replace slip (${slipLegs.length}) with these legs` : "Add to slip"}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
