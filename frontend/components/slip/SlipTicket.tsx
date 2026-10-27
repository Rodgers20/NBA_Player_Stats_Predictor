"use client";

import { useState } from 'react';
import Link from 'next/link';
import { ClipboardCopy, ImageDown, Link2, PencilLine } from 'lucide-react';
import { formatOdds, summarizeSlip, type SlipLeg, type SlipMode } from '@/lib/bet-slip.mjs';
import { useSlip } from '@/store/slip';
import { copy, copySlipText, shareSlipImage } from './slip-actions';
import { LegRow, SlipTotals } from './SlipParts';
import { SendToBook } from './SendToBook';

/** The shared slip: a ticket sized to screenshot on a phone, with the ways to send it on underneath. */
export function SlipTicket({ legs, mode, stake, code }: { legs: SlipLeg[]; mode: SlipMode; stake: number | null; code: string }) {
  const [flash, setFlash] = useState<string | null>(null);
  const load = useSlip(s => s.load);
  const say = (m: string) => { setFlash(m); setTimeout(() => setFlash(null), 2400); };
  const parlay = mode === 'parlay' && legs.length > 1;
  const american = summarizeSlip(legs).american;
  const head = parlay ? `${legs.length}-leg parlay` : legs.length === 1 ? 'Single' : `${legs.length} singles`;
  const btn = 'inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-border-card px-4 text-[14px] font-semibold text-text-soft hover:border-teal-400/60 hover:text-teal-400';

  return <div className="mx-auto flex w-full max-w-[460px] flex-col gap-4">
    <article aria-label={head} className="ql-panel overflow-hidden shadow-[0_18px_50px_rgba(0,0,0,0.45)]">
      <header className="flex items-center gap-3 border-b border-dashed border-border-card bg-ink/[.03] px-5 py-4">
        <span className="h-8 w-2 rounded-sm bg-teal-400" aria-hidden />
        <div><p className="text-xl font-black uppercase leading-none text-text-pri">Bet slip</p><p className="mt-1 text-[12px] text-text-sec">NBA · WNBA bets</p></div>
        <div className="ml-auto text-right">
          <p className="text-lg font-extrabold uppercase leading-none text-text-pri">{head}</p>
          {parlay && <p className={`mt-1 font-mono text-xl font-extrabold leading-none ${american === null ? 'text-text-sec' : 'text-lock-gold'}`}>{american === null ? 'Odds at your book' : formatOdds(american)}</p>}
        </div>
      </header>
      <ul className="divide-y divide-ink/10 px-5">{legs.map((l, i) => <LegRow key={`${l.league}${l.date}${l.player}${l.stat}`} leg={l} index={parlay ? i : undefined} />)}</ul>
      <div className="border-t border-dashed border-border-card px-5 py-4">
        {stake && <p className="mb-3 text-[13px] text-text-sec">Stake <b className="text-text-pri">${stake}</b>{mode === 'singles' && legs.length > 1 ? ' each' : ''}</p>}
        <SlipTotals legs={legs} mode={parlay || legs.length === 1 ? 'parlay' : 'singles'} stake={stake} />
      </div>
    </article>
    <div className="grid grid-cols-2 gap-2">
      <button type="button" className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-teal-400 px-4 text-[14px] font-bold text-bg-root hover:brightness-110" onClick={async () => {
        say('Making the image…');
        const r = await shareSlipImage(legs, mode, stake);
        say(r === 'failed' ? "Couldn't make the image." : r === 'saved' ? 'Image saved.' : 'Shared.');
      }}><ImageDown className="h-4 w-4" aria-hidden /> Save image</button>
      <button type="button" className={btn} onClick={async () => say(await copySlipText(legs, mode, stake) ? 'Copied. Paste it into Discord.' : "Couldn't copy.")}><ClipboardCopy className="h-4 w-4" aria-hidden /> Copy for Discord</button>
      <button type="button" className={btn} onClick={async () => say(await copy(`${window.location.origin}/slip?s=${code}`) ? 'Link copied.' : "Couldn't copy.")}><Link2 className="h-4 w-4" aria-hidden /> Copy link</button>
      <Link href="/props" className={btn} onClick={() => load(legs, mode, stake)}><PencilLine className="h-4 w-4" aria-hidden /> Edit slip</Link>
    </div>
    <SendToBook legs={legs} />
    <p className="min-h-4 text-center text-[12.5px] text-text-sec" aria-live="polite">{flash ?? ''}</p>
    <p className="text-center text-[12px] leading-relaxed text-text-sec">This app doesn&apos;t take bets. Prices are the saved quotes collected with each pick and may have moved, or come from different books; confirm them at your sportsbook. Legs from the same game tend to land or miss together, so the model chance is a rough guide.</p>
  </div>;
}
