"use client";

import { useState } from 'react';
import Link from 'next/link';
import { Clipboard, Plus, X } from 'lucide-react';
import { useSlip } from '@/store/slip';
import { formatOdds, legKey, summarizeSlip } from '@/lib/bet-slip.mjs';

export function BetSlip() {
  const { legs, remove, clear } = useSlip();
  const [stake, setStake] = useState('10');
  const [goal, setGoal] = useState('5');
  const [copyStatus, setCopyStatus] = useState('');
  const summary = summarizeSlip(legs);
  const amount = Number(stake);
  const validStake = stake.trim() !== '' && Number.isFinite(amount) && amount > 0;
  const today = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year:'numeric',month:'2-digit',day:'2-digit' }).format(new Date());
  const pastLegs = legs.some(leg => leg.date < today);
  const copyText = [
    `${legs.length}-leg basketball slip`,
    ...legs.map((leg, i) => `${i+1}. ${leg.player} — ${leg.direction} ${leg.line} ${leg.stat} (${formatOdds(leg.price)}) | ${leg.league.toUpperCase()} · ${leg.matchup} · ${leg.date}`),
    `Calculated odds: ${formatOdds(summary.american)} / ${summary.decimal?.toFixed(2) ?? '—'} decimal`,
    `Independent model estimate: ${summary.probability == null ? 'unavailable' : `${(summary.probability*100).toFixed(1)}%`}${summary.correlated ? ' — same-game correlation not modeled' : ''}`,
    'Reference slip only. Confirm selections, availability and final parlay odds at your sportsbook.',
  ].join('\n');
  async function copy() {
    try { await navigator.clipboard.writeText(copyText); setCopyStatus('Slip copied.'); }
    catch { setCopyStatus('Copy unavailable. Select the slip text below.'); }
  }
  return <aside className="ql-panel overflow-hidden border-[#59e0c8]/25" aria-label="Bet slip">
    <div className="border-b border-white/10 bg-[#59e0c8]/[.05] p-5">
      <div className="flex items-center justify-between gap-3">
        <div><p className="ql-kicker">YOUR SELECTIONS</p><h2 className="mt-1 text-xl font-bold text-[#eaf8f6]">Bet slip <span className="ml-1 text-[#59e0c8]">{legs.length}</span></h2></div>
        {!!legs.length && <button onClick={() => { clear(); setCopyStatus(''); }} className="text-xs text-[#a9bec0] hover:text-white">Clear all</button>}
      </div>
      <div className="mt-4 flex items-center justify-between gap-3 text-xs text-[#a9bec0]">
        <label htmlFor="slip-goal">Build a parlay</label>
        <select id="slip-goal" value={goal} onChange={event => setGoal(event.target.value)} className="ql-control px-2 py-1.5">
          {[2,3,4,5,6,8,10].map(n => <option key={n} value={n}>{n} legs</option>)}
        </select>
      </div>
      <div className="mt-3 h-1 overflow-hidden rounded bg-white/10" aria-hidden="true"><div className="h-full bg-[#59e0c8] transition-all" style={{width:`${Math.min(100,legs.length / Number(goal)*100)}%`}} /></div>
      <p className="mt-2 text-xs text-[#a9bec0]" role="status">{legs.length >= Number(goal) ? `${goal}-leg goal reached · keep adding or copy your slip` : `${Number(goal)-legs.length} more to your ${goal}-leg slip`}</p>
    </div>
    {!legs.length ? <div className="px-5 py-8 text-center">
      <Plus className="mx-auto mb-3 text-[#59e0c8]" size={24} />
      <p className="text-sm font-semibold text-[#eaf8f6]">Start with a prop you like</p>
      <p className="mt-2 text-xs leading-5 text-[#a9bec0]">Tap + beside a priced pick on Best Props. Your selections stay here as you browse.</p>
      <Link href="/props" className="mt-4 inline-block text-xs font-bold text-[#59e0c8]">Explore Best Props →</Link>
    </div> : <ol className="divide-y divide-white/10">
      {legs.map((leg,index) => <li key={legKey(leg)} className="px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0"><p className="text-sm font-bold text-[#eaf8f6]"><span className="mr-2 text-xs text-[#a9bec0]">{String(index+1).padStart(2,'0')}</span>{leg.player}</p>
          <p className="mt-1 text-sm font-semibold text-[#59e0c8]">{leg.direction} {leg.line} {leg.stat}</p></div>
          <button onClick={() => { remove(legKey(leg)); setCopyStatus(''); }} aria-label={`Remove ${leg.player} ${leg.stat} from slip`} className="rounded p-2 text-[#a9bec0] hover:bg-white/10 hover:text-white"><X size={16}/></button>
        </div>
        <div className="mt-2 flex justify-between gap-2 text-xs"><span className="text-[#a9bec0]">{leg.matchup} · {leg.league.toUpperCase()}</span><strong className="font-mono text-[#f5ba64]">{formatOdds(leg.price)}</strong></div>
        <p className="mt-1 text-[11px] text-[#a9bec0]">{leg.date} · Saved quote · {leg.probability == null ? 'Model probability unavailable' : `${(leg.probability*100).toFixed(1)}% model estimate`}</p>
      </li>)}
    </ol>}
    <div className="border-t border-white/10 p-5">
      <dl className="space-y-3">
        <div className="flex justify-between gap-3"><dt className="text-xs text-[#a9bec0]">Calculated combined odds</dt><dd className="font-mono text-lg font-bold text-[#f5ba64]">{formatOdds(summary.american)}</dd></div>
        <div className="flex justify-between gap-3"><dt className="text-xs text-[#a9bec0]">Decimal odds</dt><dd className="font-mono text-sm text-[#eaf8f6]">{summary.decimal?.toFixed(2) ?? '—'}</dd></div>
        <div className="flex justify-between gap-3"><dt className="text-xs text-[#a9bec0]">Est. hit chance · independent</dt><dd className="font-mono text-lg font-bold text-[#59e0c8]">{summary.probability == null ? '—' : `${(summary.probability*100).toFixed(1)}%`}</dd></div>
      </dl>
      {summary.correlated && <p className="mt-3 text-xs leading-5 text-[#f5ba64]">Same-game legs can be correlated. This percentage assumes independence; the actual joint chance and sportsbook price can differ materially.</p>}
      {pastLegs && <p className="mt-3 text-xs text-[#f5ba64]">This slip contains a past date. Replace old selections before using it.</p>}
      <label className="mt-5 flex items-center justify-between gap-3 text-xs text-[#a9bec0]">Stake ($)<input aria-label="Slip stake in dollars" type="number" min="0.01" step="0.01" value={stake} onChange={event => setStake(event.target.value)} className="ql-control w-28 px-3 py-2 text-right" /></label>
      <div className="mt-3 flex justify-between text-sm"><span className="text-[#a9bec0]">Potential return</span><strong className="text-[#eaf8f6]">{validStake && summary.decimal ? `$${(amount*summary.decimal).toFixed(2)}` : '—'}</strong></div>
      <p className="mt-1 text-right text-[11px] text-[#a9bec0]">Includes stake</p>
      <button disabled={!legs.length} onClick={() => void copy()} className="mt-5 flex w-full items-center justify-center gap-2 rounded bg-[#59e0c8] px-4 py-3 text-sm font-bold text-[#0d151b] disabled:opacity-40"><Clipboard size={16}/>Copy slip for your book</button>
      {!!legs.length && <details className="mt-3 text-xs text-[#a9bec0]"><summary className="cursor-pointer">View slip text</summary><pre className="mt-2 whitespace-pre-wrap leading-5 select-text">{copyText}</pre></details>}
      <p role="status" className="mt-2 text-xs text-[#59e0c8]">{copyStatus}</p>
      <p className="mt-3 text-[11px] leading-5 text-[#a9bec0]">Odds are the product of saved individual quotes, not a confirmed parlay offer. Quotes may come from different books. Confirm prices and availability at your book. Model estimates are unvalidated for parlays.</p>
    </div>
  </aside>;
}
