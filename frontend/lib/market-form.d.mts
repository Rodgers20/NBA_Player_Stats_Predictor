import type { TeamGame } from './team-types';

export type FormSeries = { values: number[]; line: number };
export type MarketForm = {
  moneyline: { home: FormSeries; away: FormSeries };
  spread: { home: FormSeries; away: FormSeries } | null;
  total: { home: number[]; away: number[]; line: number } | null;
};

export function marketForm(
  game: { spread: number | null; total: number | null },
  homeGames: TeamGame[],
  awayGames: TeamGame[],
  limit?: number,
): MarketForm;
