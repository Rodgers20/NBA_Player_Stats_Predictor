import type { TeamGame } from './team-types';

export type MarketSide = {
  label: string;
  value?: string | null;
  /** Recent results against today's number; omitted when there is no history. */
  form?: { values: number[]; line: number; caption: string; flat?: boolean; note?: string };
};
export type MarketPair = [MarketSide, MarketSide];

export function fmtPrice(price: number | null | undefined): string;
export function fmtLine(line: number): string;
export function gameMarkets(
  game: { spread: number | null; total: number | null; away_ml: number | null; home_ml: number | null },
  awayAbbrev: string,
  homeAbbrev: string,
  awayGames: TeamGame[],
  homeGames: TeamGame[],
): { moneyline: MarketPair | null; spread: MarketPair | null; total: MarketPair | null; totalTitle: string };
