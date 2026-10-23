"use client";

import type { MouseEvent, ReactNode } from "react";
import { usePrefs } from "@/store/prefs";
import { usePlayerSheet } from "@/store/player-sheet";
import type { AnalysisFocus } from "@/components/analysis/PlayerAnalysis";
import type { Prop } from "@/lib/types";

/** Open the analysis on this prop's market: its stat, line, side and (when quoted) price. */
export const focusFromProp = (prop: Prop): AnalysisFocus => ({
  stat: prop.stat, line: prop.line ?? null, side: /^under$/i.test(prop.direction) ? "Under" : "Over",
  price: prop.has_live_odds ? prop.price : null, book: prop.book, matchup: prop.game_matchup,
});

interface Props {
  name: string;
  /** Defaults to the league the site is showing. */
  league?: "nba" | "wnba";
  focus?: AnalysisFocus;
  className?: string;
  /** What to show instead of the name, e.g. a photo or "Analyze →". */
  children?: ReactNode;
}

/** A player's name anywhere on the site: tapping it opens their full analysis over the current page. */
export function PlayerName({ name, league, focus, className = "", children }: Props) {
  const siteLeague = usePrefs((state) => state.league);
  const open = usePlayerSheet((state) => state.open);
  function show(event: MouseEvent<HTMLButtonElement>) {
    event.stopPropagation(); // rows that expand on click should not also toggle
    const box = event.currentTarget.getBoundingClientRect();
    // Keyboard activation has no pointer position; grow from the name itself.
    const origin = event.detail ? { x: event.clientX, y: event.clientY } : { x: box.left + box.width / 2, y: box.top + box.height / 2 };
    open({ player: name, league: league ?? siteLeague, focus, origin });
  }
  return <button type="button" onClick={show} title={`Open ${name}'s analysis`}
    className={`cursor-pointer text-left hover:text-teal-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8] ${className}`}>
    {children ?? name}
  </button>;
}
