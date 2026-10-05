export interface SlipLeg {
  league: 'nba' | 'wnba'; date: string; player: string; stat: string;
  direction: string; line: number; price: number; probability: number | null;
  matchup: string; addedAt: string;
}
export function decimalOdds(price: number): number | null;
export function americanOdds(decimal: number): number | null;
export function legKey(leg: SlipLeg): string;
export function summarizeSlip(legs: SlipLeg[]): { decimal: number | null; american: number | null; probability: number | null; correlated: boolean };
export function formatOdds(price: number | null | undefined): string;

export function toggleLeg(legs: SlipLeg[], leg: SlipLeg): SlipLeg[];
