import type { MatchupProfile } from './types';

export type RankCell = { display: string; rank: number; of: number };
export type RankingRow = { label: string; defenseLabel: string; offense: RankCell; defense: RankCell; edge: 'offense' | 'defense' | 'even' };

export function offenseVsDefense(
  profiles: Record<string, MatchupProfile | null | undefined> | undefined,
  offenseTeam: string,
  defenseTeam: string,
): RankingRow[];
