"use client";
import { GameHighlights } from '@/components/games/GameHighlights';
import { usePrefs } from '@/store/prefs';
export default function HighlightsPage() {
  const league=usePrefs(state=>state.league);
  return <div className="ql-page px-4 sm:px-8 pt-8 max-w-[1440px] mx-auto pb-12"><header className="mb-7"><p className="ql-kicker">{league.toUpperCase()} / WATCH</p><h1 className="ql-heading">Highlights</h1><p className="ql-subtitle">The latest game clips and recaps, straight from ESPN.</p></header><GameHighlights league={league} recent showAll /></div>;
}
