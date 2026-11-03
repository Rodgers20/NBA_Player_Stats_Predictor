import type { NhlLine, NhlMarket, NhlMarketKey, NhlPropPlayer } from './nhl-types';
export const SORTS: { key: string; label: string }[];
export interface BoardRow { player: NhlPropPlayer; market: NhlMarket; line: NhlLine }
export function lineOptions(players: NhlPropPlayer[], market: NhlMarketKey): number[];
export function initialLine(options: number[], market: NhlMarketKey): number | null;
export function boardRows(players: NhlPropPlayer[], opts: { market: NhlMarketKey; line: number | null; date?: string; gameId?: string | number; team?: string; query?: string; sort?: string }): BoardRow[];
export function boardGames(players: NhlPropPlayer[], date?: string): { id: number; start: string; away: string; home: string }[];
