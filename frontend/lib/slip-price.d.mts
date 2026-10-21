import type { SlipLeg } from './bet-slip.mjs';

export interface PricedLeg {
  player: string; team: string; game: string; stat: string; side: 'Over' | 'Under'; prob: number; price?: number;
}
export interface SlipPriceRequest { legs: PricedLeg[] }
export interface CorrelationNote { with: number; rho: number; reason: string }
export interface SlipPrice {
  independent_prob: number; correlated_prob: number; fair_odds: number | null;
  per_leg_corr: { leg: number; notes: CorrelationNote[] }[];
}
export interface RelatedCandidate { player: string; stat: string; direction: string; line: number | null; price?: number | null }

export const MAX_PRICED_LEGS: number;
export function buildPriceRequest(legs: SlipLeg[], teamByPlayer?: Record<string, string>): SlipPriceRequest | null;
export function describeCorrelation(item: { notes: CorrelationNote[] } | undefined | null): string | null;
export function relatedProps<T extends RelatedCandidate>(leg: SlipLeg, props: T[], slipLegs: SlipLeg[], limit?: number): T[];
export function fetchSlipPrice(baseUrl: string, body: SlipPriceRequest, signal?: AbortSignal): Promise<SlipPrice>;
