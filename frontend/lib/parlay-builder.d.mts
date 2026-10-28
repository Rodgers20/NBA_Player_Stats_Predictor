import type { SlipLeg } from './bet-slip.mjs';
import type { Prop } from './types';

export type ParlayRisk = 'safe' | 'edge' | 'payout';
export function canBuildFrom(prop: Prop): boolean;
export function propLeg(prop: Prop, league: 'nba' | 'wnba', date: string): SlipLeg;
export function buildParlay<T extends Prop>(props: T[], options?: { legs?: number; risk?: ParlayRisk; include?: string[] }): T[];
