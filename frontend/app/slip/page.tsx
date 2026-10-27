"use client";

import { Suspense } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { decodeSlip } from '@/lib/bet-slip.mjs';
import { SlipTicket } from '@/components/slip/SlipTicket';

function SlipView() {
  const code = useSearchParams().get('s');
  const slip = decodeSlip(code);
  if (!code || !slip || !slip.legs.length) {
    return <div className="ql-panel mx-auto max-w-[460px] p-8 text-center">
      <h2 className="text-lg font-bold text-text-pri">This slip is empty</h2>
      <p className="mt-2 text-sm text-text-sec">Add priced picks from Best Props, then open the slip.</p>
      <Link href="/props" className="mt-4 inline-block text-sm font-bold text-teal-400">Explore Best Props →</Link>
    </div>;
  }
  return <SlipTicket legs={slip.legs} mode={slip.mode} stake={slip.stake} code={code} />;
}

/** A bet slip on its own page, ready to screenshot or share; the legs live in the link. */
export default function SlipPage() {
  return <div className="ql-page mx-auto max-w-[1440px] px-4 pb-12 pt-7 sm:px-8">
    <header className="mb-6"><p className="ql-kicker">SHARE</p><h1 className="ql-heading mt-1">Bet slip</h1>
      <p className="ql-subtitle mt-2">Screenshot it, save it as an image, or copy it for Discord.</p></header>
    <Suspense fallback={null}><SlipView /></Suspense>
  </div>;
}
