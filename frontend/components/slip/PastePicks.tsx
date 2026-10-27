"use client";

import { useRef, useState } from 'react';
import { ClipboardPaste, X } from 'lucide-react';
import { formatOdds, legKey } from '@/lib/bet-slip.mjs';
import { importPicks, type ImportedPick } from '@/lib/pick-import.mjs';
import { useSlip } from '@/store/slip';
import { useBoards } from './SendToBook';

const MAX_TEXT = 4000;

function why(pick: ImportedPick): string {
  const leg = pick.leg;
  switch (pick.status) {
    case 'exact': return `${leg!.player} ${leg!.direction} ${leg!.line} ${leg!.stat} · ${formatOdds(leg!.price)}`;
    case 'line': return `Board has ${leg!.player} ${leg!.direction} ${leg!.line} ${leg!.stat} (${formatOdds(leg!.price)}), you pasted ${pick.requested}. Tick to use the board line.`;
    case 'ambiguous': return `Which one: ${pick.candidates!.join(' or ')}? Add more of the name.`;
    case 'market': return 'That player has no line for this stat tonight.';
    default: return 'No player on tonight’s board matches.';
  }
}

/** Paste a post of picks; each is matched against tonight's board and only added once you can see what it became. */
export function PastePicks({ leagues, className }: { leagues: string[]; className?: string }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [text, setText] = useState('');
  const [results, setResults] = useState<ImportedPick[] | null>(null);
  const [chosen, setChosen] = useState<Set<number>>(new Set());
  const boards = useBoards(leagues);
  const { legs, toggle, setOpen } = useSlip();
  const ready = Object.keys(boards).length > 0;

  function match() {
    const found = importPicks(text.slice(0, MAX_TEXT), boards);
    setResults(found);
    setChosen(new Set(found.flatMap((p, i) => (p.status === 'exact' ? [i] : []))));
  }
  function add() {
    const have = new Set(legs.map(legKey));
    results?.forEach((p, i) => { if (chosen.has(i) && p.leg && !have.has(legKey(p.leg))) toggle(p.leg); });
    dialog.current?.close();
    setText(''); setResults(null);
    setOpen(true);
  }
  const flip = (i: number) => setChosen(prev => { const next = new Set(prev); if (next.has(i)) next.delete(i); else next.add(i); return next; });

  return <>
    <button type="button" onClick={() => dialog.current?.showModal()} className={className ?? 'ql-control inline-flex min-h-9 items-center gap-1.5 px-3 text-xs font-semibold text-teal-400 hover:bg-teal-400/10'}>
      <ClipboardPaste className="h-4 w-4" aria-hidden /> Paste picks
    </button>
    <dialog ref={dialog} aria-labelledby="paste-picks-title" onClose={() => setResults(null)}
      className="m-auto w-[min(520px,calc(100vw-32px))] rounded-2xl border border-border-card bg-bg-card p-0 text-text-pri backdrop:bg-black/60">
      <div className="flex items-center gap-2 border-b border-ink/10 px-5 py-3">
        <h2 id="paste-picks-title" className="text-lg font-extrabold uppercase tracking-wide">Paste picks</h2>
        <button type="button" onClick={() => dialog.current?.close()} aria-label="Close" className="-mr-2 ml-auto grid h-10 w-10 place-items-center rounded-full text-text-sec hover:bg-ink/10"><X className="h-5 w-5" aria-hidden /></button>
      </div>
      <div className="flex max-h-[70dvh] flex-col gap-3 overflow-y-auto px-5 py-4">
        <label htmlFor="paste-picks-text" className="text-[12.5px] text-text-sec">One pick per line or comma, e.g. <i>Brunson o27.5 pts, SGA u41.5 PRA, Jokic 12+ reb</i></label>
        <textarea id="paste-picks-text" value={text} maxLength={MAX_TEXT} rows={5} onChange={e => { setText(e.target.value); setResults(null); }}
          className="w-full rounded-lg border border-border-card bg-bg-root p-3 text-[14px] text-text-pri" />
        <button type="button" onClick={match} disabled={!text.trim() || !ready}
          className="min-h-11 rounded-lg border border-border-card text-[14px] font-bold text-teal-400 hover:border-teal-400/60 disabled:opacity-40">{ready ? 'Match to tonight’s board' : 'Loading tonight’s board…'}</button>
        {results && (results.length === 0
          ? <p role="status" className="text-[13px] text-text-sec">No picks found. Each needs a player, a number and a stat.</p>
          : <ul role="status" className="divide-y divide-ink/10">
            {results.map((p, i) => <li key={i} className="flex items-start gap-3 py-2.5">
              {p.leg ? <input type="checkbox" checked={chosen.has(i)} onChange={() => flip(i)} aria-label={`Add ${p.text}`} className="mt-1 h-5 w-5 shrink-0 accent-teal-400" />
                : <span className="mt-1 h-5 w-5 shrink-0 text-center text-lock-gold" aria-hidden>!</span>}
              <div className="min-w-0">
                <p className="truncate text-[12px] text-text-sec">“{p.text}”</p>
                <p className={`text-[13px] leading-snug ${p.status === 'exact' ? 'font-bold text-text-pri' : 'text-lock-gold'}`}>{why(p)}</p>
              </div>
            </li>)}
          </ul>)}
      </div>
      {results && results.length > 0 && <div className="border-t border-ink/10 px-5 py-3">
        <button type="button" onClick={add} disabled={chosen.size === 0}
          className="min-h-11 w-full rounded-lg bg-teal-400 text-[14px] font-bold text-bg-root hover:brightness-110 disabled:opacity-40">Add {chosen.size} to slip</button>
      </div>}
    </dialog>
  </>;
}
