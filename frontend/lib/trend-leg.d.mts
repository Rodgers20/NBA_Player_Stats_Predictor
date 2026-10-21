import type { TrendCard } from './types';
import type { SlipLeg } from './bet-slip.mjs';

export function trendLeg(input: { card: TrendCard; league: 'nba' | 'wnba'; date: string; price?: number | null }): SlipLeg | null;
