export type Side = "home" | "draw" | "away";
export interface Odds { home?: number | null; draw?: number | null; away?: number | null }
export interface Win { home: number; draw: number; away: number }
export function impliedProbs(odds: Odds | null | undefined): Win | null;
export function modelVsBook(
  win: Win | null | undefined,
  odds: Odds | null | undefined,
  threshold?: number,
): { side: Side; model: number; book: number; gap: number } | null;
