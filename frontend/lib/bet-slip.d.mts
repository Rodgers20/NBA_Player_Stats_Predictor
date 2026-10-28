export interface SlipLeg {
  league: 'nba' | 'wnba'; date: string; player: string; stat: string;
  direction: string; line: number; price: number; probability: number | null;
  matchup: string; addedAt: string; team?: string;
  /** True when no price was entered and the standard -110 stands in; shown with an asterisk on the slip. */
  assumedPrice?: boolean;
}
export function decimalOdds(price: number): number | null;
export function americanOdds(decimal: number): number | null;
export function legKey(leg: SlipLeg): string;
export function summarizeSlip(legs: SlipLeg[]): { decimal: number | null; american: number | null; probability: number | null; correlated: boolean };
export function formatOdds(price: number | null | undefined): string;

export function toggleLeg(legs: SlipLeg[], leg: SlipLeg): SlipLeg[];

export type SlipMode = 'parlay' | 'singles';
export const MAX_LEGS: number;
export function payout(stake: number | null, price: number | null | undefined): number | null;
export function summarizeSingles(legs: SlipLeg[], stake: number | null): { bets: number; totalStake: number | null; payout: number | null };
export function slipStakes(legs: SlipLeg[], mode: SlipMode, bankroll: { units: number; unitSize: number }, correlatedProb?: number | null): { perLeg: number[]; total: number };
export function sanitizePersistedSlip(value: unknown): { legs: SlipLeg[]; mode: SlipMode; stake: number | null; bankroll: { units: number; unitSize: number } };
export function slipText(legs: SlipLeg[], mode: SlipMode, stake: number | null): string;
export function encodeSlip(legs: SlipLeg[], mode: SlipMode, stake: number | null): string;
export function decodeSlip(code: string | null | undefined): { legs: SlipLeg[]; mode: SlipMode; stake: number | null } | null;
