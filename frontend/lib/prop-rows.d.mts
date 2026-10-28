import type { Prop } from './types';

export type RowCell = { key: string; label: string; hits: number; total: number; pct: number; tone: 'good' | 'bad' | 'neutral' };

export function searchProps<T extends Pick<Prop, 'player' | 'team' | 'opponent'>>(props: T[], query: string | undefined): T[];
export function rowCells(prop: Partial<Pick<Prop, 'hit_windows' | 'hits' | 'total' | 'hit_rate'>>): RowCell[];

export type Dash = { value: number; result: 'hit' | 'miss' | 'push' };
export type HitWindowKey = 'L5' | 'L10' | 'L20' | 'h2h' | 'season';
export function rowDashes(prop: Partial<Pick<Prop, 'line' | 'direction' | 'l10_values'>>): Dash[];
export function windowCell(prop: Partial<Pick<Prop, 'hit_windows'>>, key: string): RowCell | null;
export function sortByWindow<T extends Partial<Pick<Prop, 'hit_windows'>>>(props: T[], key: string): T[];
