import type { ChartRecord } from './types';

export type WindowKey = 5 | 10 | 20 | 'h2h' | 'home' | 'away' | 'current' | 'previous';
export type WindowContext = { nextOpponent?: string | null; currentSeason?: string; previousSeason?: string };
export type WindowCount = { key: WindowKey; n: number; hits: number; decided: number; pushes: number; rate: number | null };

export const WINDOW_KEYS: WindowKey[];
export function selectWindow<T extends Pick<ChartRecord, 'opponent' | 'season' | 'is_home'>>(games: T[], period: WindowKey, ctx?: WindowContext): T[];
export function windowCounts(games: Pick<ChartRecord, 'opponent' | 'season' | 'is_home' | 'value'>[], threshold: number, ctx?: WindowContext, side?: 'Over' | 'Under'): WindowCount[];
