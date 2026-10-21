import type { DefenseRank } from './types';
export interface DefenseRow { team: string; cells: Record<string, DefenseRank>; avgRank: number | null }
export function pivotDefense(rankings: DefenseRank[], stat: string): { positions: string[]; rows: DefenseRow[] };
export function sortRows(rows: DefenseRow[], key: string, dir?: 'asc' | 'desc'): DefenseRow[];
export function softness(rank: number, of: number): number;
