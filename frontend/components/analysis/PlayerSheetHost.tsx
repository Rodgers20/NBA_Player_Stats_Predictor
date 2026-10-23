"use client";

import { usePathname } from "next/navigation";
import { AnimatePresence } from "motion/react";
import { usePlayerSheet } from "@/store/player-sheet";
import { PlayerAnalysisSheet } from "@/components/analysis/PlayerAnalysisSheet";

const PAGE_NAMES: Record<string, string> = {
  "": "Tonight", games: "Games", props: "Props", popular: "Popular", trends: "Trends",
  teams: "Teams", bets: "My Bets", hitrates: "Hit Rates", highlights: "Highlights", performance: "Model",
};

/** Mounted once in the layout: whichever page a player name was tapped on, the same sheet opens over it. */
export function PlayerSheetHost() {
  const request = usePlayerSheet((state) => state.request);
  const close = usePlayerSheet((state) => state.close);
  const page = usePathname().split("/")[1]?.replace(/\.html$/, "") ?? "";
  return <AnimatePresence>
    {request && <PlayerAnalysisSheet key={`${request.league}-${request.player}-${request.focus?.stat ?? ""}`}
      player={request.player} league={request.league} focus={request.focus} origin={request.origin}
      backLabel={`Back to ${PAGE_NAMES[page] ?? "previous page"}`} onClose={close} />}
  </AnimatePresence>;
}
