import type { CbbPlayer, CbbPlayersData, CbbStatLine } from './cbb-types';
export interface StatSpec { key: string; label: string; title: string; digits: number; low?: boolean }
export interface BoardRow extends CbbPlayer { line: CbbStatLine; value: number; rank: number }
export const STATS: StatSpec[];
export const POSITIONS: { value: string; label: string }[];
export function statValue(line: CbbStatLine | undefined, key: string): number | null;
export function formatStat(value: number | null | undefined, key: string): string;
export function seasonLabel(season: number): string;
export function defaultSeason(payload: CbbPlayersData, minPlayers?: number, minGames?: number): number;
export function boardRows(players: CbbPlayer[], opts?: { season: number; sort?: string; dir?: 1 | -1; minGames?: number; query?: string; position?: string }): BoardRow[];
export function bestFirst(key: string): 1 | -1;
export function headshot(id: string): string;
