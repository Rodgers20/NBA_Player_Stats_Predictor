import type { Prop } from './types';

export function forBooks(prop: Prop, books: string[] | undefined): Prop | null;
export function applyBoardFilters(
  props: Prop[],
  filters?: { evOnly?: boolean; stats?: string[]; players?: string[]; books?: string[] },
): Prop[];
export function availableBooks(props: Prop[]): string[];
export function availablePlayers(props: Prop[]): string[];
export function selectedStats(stat: string | undefined): string[];
export function toggleStat(current: string | undefined, value: string): string;
