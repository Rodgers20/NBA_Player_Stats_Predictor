import type { SlipLeg } from './bet-slip.mjs';

type GameLines = { matchup: string; home_team: string; away_team: string; spread: number | null; total: number | null; home_ml: number | null; away_ml: number | null };
type Ctx = { league: 'nba' | 'wnba'; date: string };

export const ASSUMED_PRICE: number;
export function isTeamLeg(leg: Pick<SlipLeg, 'stat'> | null | undefined): boolean;
export function priceIsAssumed(leg: Pick<SlipLeg, 'stat'> | null | undefined): boolean;
export function mlLeg(input: { game: GameLines; side: 'home' | 'away'; probability?: number | null } & Ctx): SlipLeg | null;
export function spreadLeg(input: { game: GameLines; side: 'home' | 'away' } & Ctx): SlipLeg | null;
export function totalLeg(input: { game: GameLines; direction: 'Over' | 'Under' } & Ctx): SlipLeg | null;
export function legTitle(leg: Pick<SlipLeg, 'stat' | 'player' | 'line' | 'direction'>): string;
export function legDetail(leg: Pick<SlipLeg, 'stat' | 'line' | 'direction'>): string;
export function legText(leg: Pick<SlipLeg, 'stat' | 'player' | 'line' | 'direction' | 'price'>): string;
