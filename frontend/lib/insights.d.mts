import type { GameFilters } from './game-filters.mjs';
import type { Prop } from './types';

export function summarize(values: (number | null | undefined)[]): { mean: number; median: number } | null;
export function scopeLabel(filters: GameFilters): string;
export function streakInsight(input: {
  player: string;
  stat: string;
  line: number;
  games: { value: number | null }[];
  scope?: string;
  side?: 'Over' | 'Under';
}): string | null;
export function gameInsights(props: Prop[], options?: { limit?: number }): { prop: Prop; text: string }[];
