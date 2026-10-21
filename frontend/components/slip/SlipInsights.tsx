"use client";

import { formatOdds } from '@/lib/bet-slip.mjs';
import type { Bankroll } from '@/lib/bankroll.mjs';
import { bankrollAmount } from '@/lib/bankroll.mjs';
import type { SlipPrice } from '@/lib/slip-price.mjs';

const money = (v: number) => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pct = (p: number) => `${Math.round(p * 1000) / 10}%`;

/** Independent versus correlation-adjusted chance of the whole parlay, when the API could price it. */
export function CorrelationSummary({ price, independent }: { price: SlipPrice | null; independent: number | null }) {
  if (!price) return null;
  return <p className="rounded-lg bg-white/[.06] px-2.5 py-2 text-[12px] leading-snug text-[#a9bec0]">
    Independent <b className="font-mono text-[#eaf8f6]">{pct(independent ?? price.independent_prob)}</b>
    {' → '}correlation-adjusted <b className="font-mono text-[#59e0c8]">{pct(price.correlated_prob)}</b>
    {price.fair_odds != null && <> · fair odds <b className="font-mono text-[#f5ba64]">{formatOdds(price.fair_odds)}</b></>}
  </p>;
}

function Num({ label, value, min, onChange }: { label: string; value: number; min: number; onChange: (v: number) => void }) {
  return <label className="flex items-center gap-1.5">
    <span className="sr-only">{label}</span>
    <input aria-label={label} type="number" inputMode="decimal" min={min} step="any" defaultValue={value} key={value}
      onBlur={e => { const v = Number(e.target.value); if (Number.isFinite(v) && v > 0) onChange(v); else e.target.value = String(value); }}
      className="ql-control w-16 px-2 py-1 text-right font-mono text-[13px] font-semibold" />
  </label>;
}

/** Kelly suggestion for the slip total plus the bankroll it is sized from. */
export function KellyBox({ total, bankroll, onBankroll, onUse, each }: {
  total: number; bankroll: Bankroll; onBankroll: (b: Bankroll) => void; onUse: (stake: number) => void; each: boolean;
}) {
  return <div className="rounded-lg bg-white/[.06] px-2.5 py-2 text-[12px]">
    <div className="flex items-center gap-2">
      <span className="text-[#a9bec0]">Suggested stake{each ? ' (all legs)' : ''}</span>
      <b className={`ml-auto font-mono text-[17px] leading-tight ${total > 0 ? 'text-[#59e0c8]' : 'text-[#a9bec0]'}`}>{total > 0 ? money(total) : '—'}</b>
      {total > 0 && !each && <button type="button" onClick={() => onUse(total)} className="rounded-md border border-[#2b4248] px-2 py-1 text-[11.5px] font-semibold text-[#59e0c8] hover:border-[#59e0c8]/60">Use</button>}
    </div>
    <div className="mt-2 flex items-center gap-1.5 text-[#a9bec0]">
      Bankroll
      <Num label="Bankroll units" value={bankroll.units} min={1} onChange={units => onBankroll({ ...bankroll, units })} />
      units ×
      <span className="relative"><span className="pointer-events-none absolute left-1.5 top-1/2 -translate-y-1/2">$</span>
        <input aria-label="Unit size in dollars" type="number" inputMode="decimal" min={1} step="any" defaultValue={bankroll.unitSize} key={bankroll.unitSize}
          onBlur={e => { const v = Number(e.target.value); if (Number.isFinite(v) && v > 0) onBankroll({ ...bankroll, unitSize: v }); else e.target.value = String(bankroll.unitSize); }}
          className="ql-control w-16 py-1 pl-4 pr-2 text-right font-mono text-[13px] font-semibold" /></span>
      <span className="ml-auto font-mono">= {money(bankrollAmount(bankroll))}</span>
    </div>
    <p className="mt-2 text-[10.5px] leading-snug text-[#a9bec0]">Quarter Kelly from the model chance and price, capped at 5% of bankroll; none when there is no edge.</p>
  </div>;
}
