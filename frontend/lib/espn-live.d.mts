import type { BoxPlayer, BoxScore } from './boxscore-types';
import type { Game } from './types';

export type LiveState = 'pre' | 'in' | 'post';
export interface LiveGame {
  id: string; start: string | null; state: LiveState; detail: string; period: number | null; clock: string | null;
  away: { abbrev: string; score: number | null }; home: { abbrev: string; score: number | null };
}
export interface Play {
  id: string; text: string; period: number | null; clock: string; side: 'away' | 'home' | null;
  scoring: boolean; points: number; away: number | null; home: number | null; athlete: string | null;
}
export type LiveBox = BoxScore & { plays: Play[] };
export type LiveBetStatus = 'live' | 'win' | 'loss' | 'push';

export const ESPN: string;
export function parseScoreboard(league: string, json: unknown): Record<string, LiveGame>;
export function withLive<T extends Game>(game: T, live: LiveGame | undefined): T;
export function parseSummary(league: string, json: unknown): LiveBox;
export function statValue(stats: Record<string, string>, stat: string): number | null;
export function nameKey(name: string): string;
export function findPlayer(box: BoxScore | undefined, name: string): BoxPlayer | null;
export function liveBet(bet: { side: string; line: number }, value: number, state: string): { value: number; status: LiveBetStatus; need: number };
export function summaryPoll(state: string | undefined): number | false;
export function scoreboardPoll(games: { state: string }[]): number | false;
