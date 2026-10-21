import type { ChartRecord } from './types';
export const WINDOW_KEYS: readonly ['L5', 'L10', 'L20', 'H2H', 'Season'];
export type WindowKey = (typeof WINDOW_KEYS)[number];
export function windowGames(games: ChartRecord[], key: WindowKey, opponent?: string): ChartRecord[];
export function gradeGames(games: ChartRecord[], line: number | null, direction?: string): ChartRecord[];
export function hitSummary(graded: ChartRecord[]): { hits: number; total: number };
export function average(games: ChartRecord[]): number | null;
