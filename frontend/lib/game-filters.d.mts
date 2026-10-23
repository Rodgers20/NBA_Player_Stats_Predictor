import type { ChartRecord } from './types';

export type RestFilter = 'any' | '0' | '1' | '2' | '3+';
export type GameFilters = {
  homeOnly: boolean;
  awayOnly: boolean;
  opponent: string;
  season: string;
  rest: RestFilter;
  withPlayer: string;
  withoutPlayer: string;
  margin: [number, number];
  result: 'any' | 'W' | 'L';
  minutes: [number, number] | null;
};
type GameContext = Pick<ChartRecord, 'is_home' | 'minutes' | 'margin' | 'won' | 'rest' | 'out' | 'opponent' | 'season'>;
export function opponentOf(game: Pick<ChartRecord, 'opponent'>): string;
export function opponentOptions(games: Pick<ChartRecord, 'opponent'>[]): string[];

export const MARGIN_CAP: number;
export const DEFAULT_GAME_FILTERS: GameFilters;
export function activeFilterCount(filters: GameFilters): number;
export function applyGameFilters<T extends GameContext>(games: T[], filters?: GameFilters): T[];
export function teammateOptions(games: Pick<ChartRecord, 'out'>[]): string[];
export function histogram(values: (number | null | undefined)[], low: number, high: number, bins: number): number[];
export function minutesRange(games: Pick<ChartRecord, 'minutes'>[]): [number, number];
