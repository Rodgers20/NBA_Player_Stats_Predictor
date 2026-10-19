"use client";

import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { ClipboardCopy, ExternalLink, ImageDown, Ticket, X } from 'lucide-react';
import { legKey, summarizeSlip } from '@/lib/bet-slip.mjs';
import { useSlip } from '@/store/slip';
import { copySlipText, shareSlipImage, slipPath } from './slip-actions';
import { LegRow, ModeSwitch, SlipTotals, StakeInput } from './SlipParts';

const EASE = [0.4, 0, 0.2, 1] as const;
const today = () => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());

/**
 * The bet slip: a pill in the corner once a leg is on it, opening to a panel (a sheet from the
 * bottom on a phone). Mounted once in the layout, so legs follow you between pages.
 */
export function FloatingSlip() {
  const pathname = usePathname();
  const reduced = useReducedMotion();
  const { legs, mode, stake, open, added, remove, clear, setMode, setStake, setOpen } = useSlip();
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, setOpen]);
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), 2200);
    return () => clearTimeout(t);
  }, [flash]);

  // The slip's own page already shows it in full.
  if (pathname?.startsWith('/slip')) return null;
  const show = legs.length > 0;
  const effMode = legs.length > 1 ? mode : 'parlay';
  const singles = effMode === 'singles';
  const correlated = effMode === 'parlay' && summarizeSlip(legs).correlated && legs.length > 1;
  const stale = legs.some(l => l.date < today());

  return <>
    <AnimatePresence>
      {show && !open && <motion.button key="pill" type="button" onClick={() => setOpen(true)}
        initial={reduced ? false : { opacity: 0, y: 16, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={reduced ? undefined : { opacity: 0, y: 16 }}
        transition={{ duration: 0.2, ease: EASE }} aria-label={`Open bet slip, ${legs.length} ${legs.length === 1 ? 'leg' : 'legs'}`}
        className="fixed bottom-[calc(4.75rem+env(safe-area-inset-bottom))] right-4 z-[55] flex min-h-12 items-center gap-2.5 rounded-full border border-[#59e0c8]/50 bg-[#152128] py-2 pl-3 pr-4 shadow-[0_10px_30px_rgba(0,0,0,0.45)] hover:border-[#59e0c8] min-[901px]:bottom-6 min-[901px]:right-8">
        <motion.span key={added} initial={reduced ? false : { scale: 1.35 }} animate={{ scale: 1 }} transition={{ duration: 0.25, ease: EASE }}
          className="grid h-7 min-w-7 place-items-center rounded-full bg-[#59e0c8] px-1.5 text-[13px] font-extrabold text-[#0d151b]">{legs.length}</motion.span>
        <span className="text-[15px] font-extrabold uppercase tracking-wide text-[#eaf8f6]">Bet slip</span>
      </motion.button>}
    </AnimatePresence>

    <AnimatePresence>
      {show && open && <>
        <motion.div key="scrim" className="fixed inset-0 z-[58] bg-black/60 min-[901px]:bg-black/25" onClick={() => setOpen(false)}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} aria-hidden />
        <motion.section key="panel" role="dialog" aria-modal="true" aria-labelledby="bet-slip-title"
          initial={reduced ? false : { opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} exit={reduced ? undefined : { opacity: 0, y: 40 }} transition={{ duration: 0.24, ease: EASE }}
          className="fixed inset-x-0 bottom-0 z-[60] flex max-h-[88dvh] flex-col rounded-t-2xl border border-[#2b4248] bg-[#152128] shadow-[0_-12px_40px_rgba(0,0,0,0.5)] min-[901px]:inset-x-auto min-[901px]:bottom-6 min-[901px]:right-8 min-[901px]:max-h-[min(760px,calc(100dvh-48px))] min-[901px]:w-[400px] min-[901px]:rounded-2xl">
          <div className="mx-auto mt-2 h-1 w-10 rounded-full bg-[#2b4248] min-[901px]:hidden" aria-hidden />
          <header className="flex items-center gap-3 px-5 pb-3 pt-3 min-[901px]:pt-4">
            <Ticket className="h-5 w-5 text-[#59e0c8]" aria-hidden />
            <h2 id="bet-slip-title" className="text-xl font-extrabold uppercase tracking-wide text-[#eaf8f6]">Bet slip</h2>
            <span className="rounded-full bg-white/10 px-2 py-0.5 text-[12px] font-semibold text-[#a9bec0]">{legs.length}</span>
            <button type="button" onClick={() => clear()} className="ml-auto text-[12.5px] text-[#a9bec0] hover:text-white">Clear</button>
            <button type="button" onClick={() => setOpen(false)} aria-label="Close bet slip" className="-mr-2 grid h-10 w-10 place-items-center rounded-full text-[#a9bec0] hover:bg-white/10 hover:text-white"><X className="h-5 w-5" aria-hidden /></button>
          </header>
          {legs.length > 1 && <div className="px-5 pb-2"><ModeSwitch mode={mode} onChange={setMode} legs={legs.length} /></div>}
          <ul className="min-h-0 flex-1 divide-y divide-white/10 overflow-y-auto overscroll-contain px-5">
            <AnimatePresence initial={false}>
              {legs.map((l, i) => <motion.div key={legKey(l)} layout={!reduced} initial={reduced ? false : { opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }}
                exit={reduced ? undefined : { opacity: 0, x: -12, height: 0 }} transition={{ duration: 0.18, ease: EASE }}>
                <LegRow leg={l} index={legs.length > 1 && !singles ? i : undefined} onRemove={() => remove(legKey(l))} />
              </motion.div>)}
            </AnimatePresence>
          </ul>
          <footer className="flex flex-col gap-3 border-t border-white/10 px-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] pt-4">
            <StakeInput stake={stake} onChange={setStake} each={singles} />
            <SlipTotals legs={legs} mode={effMode} stake={stake} />
            {correlated && <p className="text-[11.5px] leading-snug text-[#f5ba64]">Same-game legs can be correlated; the model chance assumes they are not.</p>}
            {stale && <p className="text-[11.5px] leading-snug text-[#f5ba64]">This slip has a past date. Replace old selections before using it.</p>}
            <div className="grid grid-cols-[1fr_auto_auto] gap-2">
              <a href={slipPath(legs, effMode, stake)} target="_blank" rel="noopener" className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-[#59e0c8] px-4 text-[14px] font-bold text-[#0d151b] hover:brightness-110">Open slip <ExternalLink className="h-4 w-4" aria-hidden /></a>
              <button type="button" title="Copy for Discord" aria-label="Copy slip as text for Discord" onClick={async () => setFlash(await copySlipText(legs, effMode, stake) ? 'Copied. Paste it into Discord.' : "Couldn't copy.")}
                className="grid min-h-11 w-11 place-items-center rounded-lg border border-[#2b4248] text-[#a9bec0] hover:border-[#59e0c8]/60 hover:text-[#59e0c8]"><ClipboardCopy className="h-[18px] w-[18px]" aria-hidden /></button>
              <button type="button" title="Save as image" aria-label="Save slip as an image" onClick={async () => {
                setFlash('Making the image…');
                const r = await shareSlipImage(legs, effMode, stake);
                setFlash(r === 'failed' ? "Couldn't make the image." : r === 'saved' ? 'Image saved.' : null);
              }} className="grid min-h-11 w-11 place-items-center rounded-lg border border-[#2b4248] text-[#a9bec0] hover:border-[#59e0c8]/60 hover:text-[#59e0c8]"><ImageDown className="h-[18px] w-[18px]" aria-hidden /></button>
            </div>
            <p className={`min-h-4 text-center text-[12px] ${flash ? 'text-[#a9bec0]' : 'text-[#a9bec0]/70'}`} aria-live="polite">{flash ?? 'Open slip gives a page to screenshot or share. Prices are saved quotes, not a confirmed parlay offer.'}</p>
          </footer>
        </motion.section>
      </>}
    </AnimatePresence>
  </>;
}
