import type { Prop, BookQuote, HitWindows } from './types';
export type Tone = 'good' | 'bad' | 'neutral';
export function formatOdds(value: number | null | undefined): string | null;
export function edgeTier(edge: number | null | undefined): 0 | 1 | 2 | 3 | null;
export function hitTone(pct: number | null | undefined): Tone;
export function hitChips(windows: HitWindows | null | undefined): { key: string; label: string; hits: number; total: number; pct: number; tone: Tone }[];
export function bestBookKeys(books: BookQuote[] | undefined, side: 'over' | 'under'): Set<string>;
export function filterMinStars(props: Prop[], minStars: number): Prop[];
export function sortByEdge(props: Prop[]): Prop[];
