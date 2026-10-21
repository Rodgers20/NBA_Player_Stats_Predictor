export type Side = 'Over' | 'Under';
export type SideResult = 'hit' | 'miss' | 'push';

export function gradeSide(value: number, line: number, side: Side): SideResult;
export function hitSummary(values: number[], line: number, side?: Side, lastN?: number): {
  n: number; hits: number; misses: number; pushes: number; rate: number | null;
  recent: { value: number; result: SideResult }[];
};
export function gradeBet(
  bet: { game_date: string; side: string; line: number },
  games: { date: string; value: number | null }[],
): 'win' | 'loss' | 'push' | null;
