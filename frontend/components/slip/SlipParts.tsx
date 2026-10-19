"use client";

import { X } from 'lucide-react';
import { formatOdds, summarizeSingles, summarizeSlip, payout, type SlipLeg, type SlipMode } from '@/lib/bet-slip.mjs';

const money = (v: number) => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

/** One leg: who, the pick, the saved price, and what backs it. */
export function LegRow({ leg, index, onRemove }: { leg: SlipLeg; index?: number; onRemove?: () => void }) {
  return <li className="flex items-start gap-3 py-3">
    {index !== undefined && <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-white/10 text-[11px] font-bold text-[#a9bec0]">{index + 1}</span>}
    <div className="min-w-0 flex-1">
      <p className="truncate text-sm font-bold text-[#eaf8f6]">{leg.player}</p>
      <p className="text-[15px] font-bold text-[#59e0c8]">{leg.direction} {leg.line} {leg.stat}</p>
      <p className="mt-0.5 text-[11px] text-[#a9bec0]">{leg.league.toUpperCase()} · {leg.matchup} · {leg.date}{leg.probability != null && ` · Model ${(leg.probability * 100).toFixed(1)}%`}</p>
    </div>
    <div className="flex shrink-0 items-center gap-1">
      <span className="font-mono text-sm font-bold text-[#f5ba64]">{formatOdds(leg.price)}</span>
      {onRemove && <button type="button" onClick={onRemove} aria-label={`Remove ${leg.player} ${leg.stat} from slip`} className="-mr-1.5 grid h-9 w-9 place-items-center rounded-full text-[#a9bec0] hover:bg-white/10 hover:text-white"><X size={16} aria-hidden /></button>}
    </div>
  </li>;
}

function Fig({ label, value, accent, sub }: { label: string; value: string; accent?: boolean; sub?: string }) {
  return <div className="rounded-lg bg-white/[.06] px-2.5 py-2">
    <dt className="text-[#a9bec0]">{label}</dt>
    <dd className={`font-mono text-[17px] font-bold leading-tight ${accent ? 'text-[#59e0c8]' : 'text-[#eaf8f6]'}`}>{value}</dd>
    {sub && <dd className="mt-0.5 text-[10.5px] leading-tight text-[#a9bec0]">{sub}</dd>}
  </div>;
}

/** Odds, payout and the model's chance, for the parlay or for the singles together. */
export function SlipTotals({ legs, mode, stake }: { legs: SlipLeg[]; mode: SlipMode; stake: number | null }) {
  if (mode === 'singles') {
    const t = summarizeSingles(legs, stake ?? 0);
    return <dl className="grid grid-cols-3 gap-2 text-[12px]">
      <Fig label="Bets" value={String(t.bets)} />
      <Fig label="Total stake" value={t.totalStake ? money(t.totalStake) : '—'} />
      <Fig label="Payout, all win" value={t.payout ? money(t.payout) : '—'} accent={!!t.payout} />
    </dl>;
  }
  const s = summarizeSlip(legs);
  const win = stake ? payout(stake, s.american) : null;
  return <dl className="grid grid-cols-3 gap-2 text-[12px]">
    <Fig label={legs.length === 1 ? 'Odds' : 'Parlay odds'} value={formatOdds(s.american)} accent={s.american !== null} />
    <Fig label="Payout" value={win ? money(win) : '—'} sub={win ? undefined : 'Enter a stake'} />
    <Fig label={legs.length === 1 ? 'Model chance' : 'Model, all hit'} value={s.probability === null ? '—' : `${Math.round(s.probability * 1000) / 10}%`} sub="Assumes independence" />
  </dl>;
}

/** Parlay or singles. */
export function ModeSwitch({ mode, onChange, legs }: { mode: SlipMode; onChange: (m: SlipMode) => void; legs: number }) {
  return <div role="group" aria-label="Bet type" className="grid grid-cols-2 rounded-lg bg-[#0d151b] p-[3px]">
    {(['parlay', 'singles'] as const).map(m => <button key={m} type="button" aria-pressed={mode === m} onClick={() => onChange(m)}
      className={`min-h-9 rounded-md text-[13px] font-semibold transition-colors ${mode === m ? 'bg-[#59e0c8] text-[#0d151b]' : 'text-[#a9bec0] hover:text-white'}`}>
      {m === 'parlay' ? `Parlay · ${legs} legs` : `Singles · ${legs}`}</button>)}
  </div>;
}

export function StakeInput({ stake, onChange, each }: { stake: number | null; onChange: (v: number | null) => void; each: boolean }) {
  return <label className="flex items-center gap-3 text-[13px]">
    <span className="shrink-0 text-[#a9bec0]">Stake{each ? ' each' : ''}</span>
    <span className="relative flex-1">
      <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#a9bec0]">$</span>
      <input aria-label="Slip stake in dollars" type="number" inputMode="decimal" min={0} step={1} placeholder="0" value={stake ?? ''}
        onChange={e => { const v = Number(e.target.value); onChange(e.target.value === '' || !Number.isFinite(v) || v <= 0 ? null : v); }}
        className="ql-control w-full pl-7 pr-3 text-right text-[15px] font-semibold" />
    </span>
  </label>;
}
