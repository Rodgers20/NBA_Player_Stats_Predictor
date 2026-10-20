import type { Prop } from './types';

export type RowCell = { key: string; label: string; hits: number; total: number; pct: number; tone: 'good' | 'bad' | 'neutral' };

export function searchProps<T extends Pick<Prop, 'player' | 'team' | 'opponent'>>(props: T[], query: string | undefined): T[];
export function rowCells(prop: Partial<Pick<Prop, 'hit_windows' | 'hits' | 'total' | 'hit_rate'>>): RowCell[];
