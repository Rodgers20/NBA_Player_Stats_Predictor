"use client";

import { useState } from 'react';
import { useQueries } from '@tanstack/react-query';
import { ExternalLink } from 'lucide-react';
import { api } from '@/lib/api';
import { formatOdds, legKey, type SlipLeg } from '@/lib/bet-slip.mjs';
import { BOOKS, handoff, type BookKey, type LegHandoff } from '@/lib/sportsbook-links.mjs';
import type { PropsResponse } from '@/lib/types';

const BOOK_KEY = 'slip-book';

/** Tonight's full board per league, on the same cache key the Props page's builder uses. */
export function useBoards(leagues: string[]): Record<string, PropsResponse> {
  const unique = [...new Set(leagues)];
  const results = useQueries({ queries: unique.map(league => {
    const params = { league, include_research: true, limit: 500 };
    return { queryKey: ['props', params], queryFn: () => api.props(params), staleTime: 60_000 };
  }) });
  return Object.fromEntries(unique.flatMap((league, i) => (results[i]?.data ? [[league, results[i].data]] : [])));
}

function note(h: LegHandoff, book: string): { text: string; warn: boolean } {
  const opens = h.depth === 'slip' ? 'opens in the bet slip' : h.depth === 'game' ? 'opens the game page' : `opens ${book}; search the player there`;
  switch (h.status) {
    case 'ready': return { text: `Same line and price, ${opens}`, warn: false };
    case 'price': return { text: `Price is now ${formatOdds(h.price)}, ${opens}`, warn: true };
    case 'line': return { text: `${book} has ${h.line}${h.price != null ? ` (${formatOdds(h.price)})` : ''}, not this line. ${h.depth === 'game' ? 'Find it on the game page.' : 'Search the player there.'}`, warn: true };
    case 'book': return { text: `${book} has no quote for this. Search it there.`, warn: true };
    default: return { text: `Not on tonight's board. Search it at ${book}.`, warn: true };
  }
}

/** One tap puts every linked leg in the book's bet slip, where the book builds the parlay; always shown, disabled with the reason when no leg can go. */
function AddAll({ combined, linked, total, name }: { combined: string | null; linked: number; total: number; name: string }) {
  const label = total === 1 ? `Add to ${name} bet slip` : linked === total ? `Add all ${total} legs to ${name} as a parlay` : `Add ${linked} of ${total} legs to ${name} bet slip`;
  const cls = 'mt-3 inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-lg px-4 text-[15px] font-extrabold';
  if (!combined) return <>
    <button type="button" disabled className={`${cls} cursor-not-allowed bg-ink/10 text-text-sec`}>{total === 1 ? `Add to ${name} bet slip` : `Add all ${total} legs to ${name}`}</button>
    <p className="mt-1.5 text-[11.5px] leading-snug text-lock-gold">{name} hasn&apos;t sent bet-slip links for these lines yet. They arrive with the next quote refresh; until then use Open on each leg.</p>
  </>;
  return <>
    <a href={combined} target="_blank" rel="noopener noreferrer" className={`${cls} bg-teal-400 text-bg-root hover:brightness-110`}>{label} <ExternalLink className="h-4 w-4" aria-hidden /></a>
    {linked < total && <p className="mt-1.5 text-[11.5px] leading-snug text-lock-gold">The other {total - linked} {total - linked === 1 ? 'leg needs' : 'legs need'} adding by hand: see below.</p>}
  </>;
}

/** Hand the slip to a sportsbook: the deepest link per leg, and what to check before betting. Nothing is placed here. */
export function SendToBook({ legs }: { legs: SlipLeg[] }) {
  // The slip only renders once its persisted legs hydrate in the browser, so storage is readable here.
  const [book, setBook] = useState<BookKey>(() => {
    try { const saved = localStorage.getItem(BOOK_KEY); return saved === 'draftkings' ? saved : 'fanduel'; } catch { return 'fanduel'; }
  });
  const choose = (next: BookKey) => { setBook(next); try { localStorage.setItem(BOOK_KEY, next); } catch { /* not remembered */ } };

  const boards = useBoards(legs.map(l => l.league));
  const { legs: handoffs, combined, linked } = handoff(legs, boards, book);
  const name = BOOKS[book].name;

  return <section aria-labelledby="send-to-book" className="rounded-xl border border-border-card p-3">
    <div className="flex items-center gap-2">
      <h3 id="send-to-book" className="text-[13px] font-extrabold uppercase tracking-wide text-text-pri">Send to sportsbook</h3>
      <div role="radiogroup" aria-label="Sportsbook" className="ml-auto flex rounded-lg bg-ink/[.06] p-0.5">
        {(Object.keys(BOOKS) as BookKey[]).map(key => <button key={key} type="button" role="radio" aria-checked={book === key} onClick={() => choose(key)}
          className={`min-h-9 rounded-md px-2.5 text-[12px] font-bold ${book === key ? 'bg-teal-400 text-bg-root' : 'text-text-sec hover:text-ink'}`}>{BOOKS[key].name}</button>)}
      </div>
    </div>
    <AddAll combined={combined} linked={linked} total={legs.length} name={name} />
    <ul className="mt-2 divide-y divide-ink/10">
      {legs.map((leg, i) => {
        const h = handoffs[i];
        const n = note(h, name);
        return <li key={legKey(leg)} className="flex items-center gap-2 py-2">
          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] font-bold text-text-pri">{leg.player} {leg.direction} {leg.line} {leg.stat}</p>
            <p className={`text-[11px] leading-snug ${n.warn ? 'text-lock-gold' : 'text-text-sec'}`}>{n.text}</p>
          </div>
          {h.url && <a href={h.url} target="_blank" rel="noopener noreferrer" aria-label={`Open ${leg.player} ${leg.stat} at ${name}`}
            className="inline-flex min-h-9 shrink-0 items-center gap-1 rounded-lg border border-border-card px-2.5 text-[12px] font-bold text-teal-400 hover:border-teal-400/60">Open <ExternalLink className="h-3.5 w-3.5" aria-hidden /></a>}
        </li>;
      })}
    </ul>
    <p className="mt-2 text-[11px] leading-snug text-text-sec">Check every line and price at {name} before you bet. Links open the book; nothing is placed from here.</p>
  </section>;
}
