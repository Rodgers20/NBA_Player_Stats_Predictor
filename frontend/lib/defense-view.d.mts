import type { DefenseRank } from './types';
export type Grade = 'A' | 'B' | 'C' | 'D' | 'F';
export type DefStat = 'PTS' | 'REB' | 'AST';
export const DEF_STATS: readonly DefStat[];
export interface PositionCell extends DefenseRank { leagueAvg: number; grade: Grade }
export interface TeamDefense {
  byStat: Partial<Record<DefStat, { rank: number; of: number; grade: Grade }>>;
  softest: DefenseRank;
  toughest: DefenseRank;
}
export function gradeFor(rank: number, of: number): Grade;
export function basePosition(position: string | null | undefined): 'G' | 'F' | 'C' | null;
export function positionMatchup(rankings: DefenseRank[], team: string, position: string): Partial<Record<DefStat, PositionCell>>;
export function teamDefense(rankings: DefenseRank[], team: string): TeamDefense | null;
