"use client";

import { ExternalLink, Minus, Plus } from 'lucide-react';
import type { SlipLeg } from '@/lib/bet-slip.mjs';
import { modifyLeg } from '@/lib/slip-modify.mjs';
import type { LegHandoff } from '@/lib/sportsbook-links.mjs';
import type { Prop } from '@/lib/types';

const STEP = 1;
const ctl = 'grid min-h-9 place-items-center rounded-md border border-border-card text-text-sec hover:border-teal-400/60 hover:text-ink';

/** Change a leg's side or line in place (it stays on the parlay), or bet it on its own at the chosen book. */
export function ModifyLeg({ leg, prop, handoff, book, onChange }: { leg: SlipLeg; prop?: Prop; handoff?: LegHandoff; book: string; onChange: (leg: SlipLeg) => void }) {
  const change = (c: { direction?: string; line?: number }) => onChange(modifyLeg(leg, prop, c));
  const atSlip = handoff?.depth === 'slip';
  return <div className="-mt-1 mb-3 flex flex-col gap-2 rounded-lg bg-ink/[.04] p-2.5">
    <div className="flex items-center gap-2">
      <div role="radiogroup" aria-label={`${leg.player} side`} className="grid flex-1 grid-cols-2 rounded-md bg-bg-root p-0.5">
        {(['Over', 'Under'] as const).map(side => <button key={side} type="button" role="radio" aria-checked={leg.direction === side} onClick={() => change({ direction: side })}
          className={`min-h-8 rounded text-[12.5px] font-bold ${leg.direction === side ? 'bg-teal-400 text-bg-root' : 'text-text-sec hover:text-ink'}`}>{side}</button>)}
      </div>
      <button type="button" aria-label="Lower line" onClick={() => change({ line: leg.line - STEP })} className={`${ctl} w-9`}><Minus size={15} aria-hidden /></button>
      <span className="w-12 text-center font-mono text-[15px] font-bold tabular-nums text-text-pri" aria-live="polite">{leg.line}</span>
      <button type="button" aria-label="Raise line" onClick={() => change({ line: leg.line + STEP })} className={`${ctl} w-9`}><Plus size={15} aria-hidden /></button>
    </div>
    {handoff?.url && <a href={handoff.url} target="_blank" rel="noopener noreferrer"
      className="inline-flex min-h-9 items-center justify-center gap-1.5 rounded-md border border-teal-400/50 text-[12.5px] font-bold text-teal-400 hover:border-teal-400">
      {atSlip ? `Bet this one alone at ${book}` : `Find it at ${book}`} <ExternalLink className="h-3.5 w-3.5" aria-hidden /></a>}
    {handoff && !atSlip && <p className="text-[11px] leading-snug text-lock-gold">{handoff.status === 'line' ? `${book} lists ${handoff.line} for this, so it can't go straight into the slip at ${leg.line}.` : `No ${book} slip link for this leg yet.`}</p>}
  </div>;
}
