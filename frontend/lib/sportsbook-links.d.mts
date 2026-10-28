import type { SlipLeg } from './bet-slip.mjs';
import type { Prop } from './types';

export type BookKey = 'fanduel' | 'draftkings' | 'fanatics';
export const BOOKS: Record<BookKey, { name: string; home: string }>;
export interface LegHandoff {
  status: 'ready' | 'price' | 'line' | 'book' | 'board' | 'quarter';
  depth: 'slip' | 'game' | 'home';
  url: string | null;
  line: number | null;
  price: number | null;
}
export function safeBookUrl(url: unknown, bookKey: string): string | null;
export function handoffLeg(leg: SlipLeg, board: { props?: Prop[] } | null | undefined, bookKey: BookKey): LegHandoff;
export const FANDUEL_STATES: string[];
export interface QuarterQuote { player: string; stat: string; line: number; books: import('./types').BookQuote[]; game_matchup?: string | null }
export function boardWithQuarterQuotes<B extends { props?: Prop[] }>(board: B | null | undefined, quotes: QuarterQuote[] | null | undefined): B & { props: Prop[] };
export function handoff(legs: SlipLeg[], boards: Partial<Record<string, { props?: Prop[] }>>, bookKey: BookKey, state?: string | null): { legs: LegHandoff[]; combined: string | null; linked: number };
