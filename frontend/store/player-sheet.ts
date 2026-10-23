import { create } from "zustand";
import type { AnalysisFocus } from "@/components/analysis/PlayerAnalysis";

export interface PlayerSheetRequest {
  player: string;
  league: "nba" | "wnba";
  /** The prop it was opened from, when there is one: its line and side start the analysis. */
  focus?: AnalysisFocus;
  /** Where the tap happened, so the sheet grows out of it. */
  origin?: { x: number; y: number } | null;
}

/** One full-screen player analysis for the whole app, opened from any player name. */
export const usePlayerSheet = create<{
  request: PlayerSheetRequest | null;
  open: (request: PlayerSheetRequest) => void;
  close: () => void;
}>((set) => ({
  request: null,
  open: (request) => set({ request }),
  close: () => set({ request: null }),
}));
