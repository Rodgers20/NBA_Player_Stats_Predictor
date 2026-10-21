export const KELLY_FRACTION = 0.25;
export const MAX_STAKE_FRACTION = 0.05;
export const DEFAULT_BANKROLL = { units: 100, unitSize: 10 };

/** Fraction of bankroll to stake: fractional Kelly, capped; 0 without a positive edge or valid inputs. */
export function kellyFraction(probability, price, { fraction = KELLY_FRACTION, cap = MAX_STAKE_FRACTION } = {}) {
  if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) return 0;
  if (!Number.isFinite(price) || (price > -100 && price < 100)) return 0;
  const b = price > 0 ? price / 100 : 100 / -price;
  const full = (b * probability - (1 - probability)) / b;
  return full > 0 ? Math.min(full * fraction, cap) : 0;
}

export function bankrollAmount(bankroll) {
  const amount = Number(bankroll?.units) * Number(bankroll?.unitSize);
  return Number.isFinite(amount) && amount > 0 ? amount : 0;
}

/** Suggested stake in dollars (rounded to cents). */
export function suggestedStake(probability, price, bankroll, options) {
  return Math.round(bankrollAmount(bankroll) * kellyFraction(probability, price, options) * 100) / 100;
}

/** Coerces untrusted (stored or typed) bankroll values to positive finite numbers, else the default. */
export function sanitizeBankroll(value) {
  const units = Number(value?.units), unitSize = Number(value?.unitSize);
  return Number.isFinite(units) && units > 0 && Number.isFinite(unitSize) && unitSize > 0
    ? { units, unitSize } : DEFAULT_BANKROLL;
}
