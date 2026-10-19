export interface BreakdownRow {
  key: string; bets: number; wins: number; losses: number; pushes: number;
  staked: number; profit: number; roi: number | null;
}
export interface CurvePoint { n: number; date: string; pnl: number }
export interface JournalSummary {
  count: number; settled: number; profit: number; roi: number | null; pending: number; staked: number;
  by_stat: BreakdownRow[]; by_side: BreakdownRow[]; by_book: BreakdownRow[]; by_odds: BreakdownRow[];
  curve: CurvePoint[]; max_drawdown: number;
}
export const ODDS_RANGES: string[];
export function oddsRange(price: number): string;
export function summarizeBets(bets: unknown[]): JournalSummary;
export function localJournalList(storage: Storage, mode: string): unknown;
export function localJournalAdd(storage: Storage, entry: Record<string, unknown>): unknown;
export function localJournalSettle(storage: Storage, id: string, result: string): unknown;
