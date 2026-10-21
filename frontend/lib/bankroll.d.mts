export interface Bankroll { units: number; unitSize: number }
export interface KellyOptions { fraction?: number; cap?: number }
export const KELLY_FRACTION: number;
export const MAX_STAKE_FRACTION: number;
export const DEFAULT_BANKROLL: Bankroll;
export function kellyFraction(probability: number | null | undefined, price: number | null | undefined, options?: KellyOptions): number;
export function bankrollAmount(bankroll: Partial<Bankroll> | null | undefined): number;
export function suggestedStake(probability: number | null | undefined, price: number | null | undefined, bankroll: Partial<Bankroll>, options?: KellyOptions): number;
export function sanitizeBankroll(value: unknown): Bankroll;
