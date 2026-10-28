import type { SlipLeg } from './bet-slip.mjs';
import type { Prop } from './types';

export interface ParsedPick { query: string; direction: 'Over' | 'Under'; line: number; stat: string; assumedDirection: boolean }
export interface ImportedPick {
  text: string;
  status: 'exact' | 'line' | 'ambiguous' | 'player' | 'market';
  leg?: SlipLeg;
  /** The line the post asked for, when the board only has a different one. */
  requested?: number;
  candidates?: string[];
}
export function splitPicks(text: string): string[];
export function parsePick(text: string): ParsedPick | null;
export function importPicks(text: string, boards: Partial<Record<string, { target_date: string | null; props?: Prop[] }>>): ImportedPick[];
