import type { Prop } from './types';

export function breakEven(price: number | null | undefined): number | null;
export function quoteAge(updatedAt: string | null | undefined, now?: number): { label: string; stale: boolean } | null;
export function recordFor(
  record: { by_stat?: Record<string, { hit: number; total: number; pct: number }> } | null | undefined,
  stat: string,
): { hit: number; total: number; pct: number; early: boolean } | null;
export function playReason(prop: Prop): string | null;
export function tonightBoard(props: Prop[]): { picks: Prop[]; leans: Prop[]; research: Prop[] };
