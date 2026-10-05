export function decimalOdds(price) {
  return Number.isFinite(price) && Math.abs(price) >= 100 ? (price > 0 ? 1 + price / 100 : 1 + 100 / -price) : null;
}
export function americanOdds(decimal) {
  if (!Number.isFinite(decimal) || decimal <= 1) return null;
  return decimal >= 2 ? Math.round((decimal - 1) * 100) : Math.round(-100 / (decimal - 1));
}
export function legKey(leg) {
  return JSON.stringify([leg.league, leg.date, leg.player, leg.stat]);
}
export function summarizeSlip(legs) {
  if (!legs.length) return { decimal: null, american: null, probability: null, correlated: false };
  const prices = legs.map(leg => decimalOdds(leg.price));
  const decimal = prices.every(price => price !== null) ? prices.reduce((a, b) => a * b, 1) : null;
  const games = legs.map(leg => `${leg.league}:${leg.date}:${leg.matchup}`);
  const correlated = new Set(games).size !== games.length || legs.some(leg => !leg.matchup);
  const probability = legs.every(leg => Number.isFinite(leg.probability) && leg.probability > 0 && leg.probability < 1)
    ? legs.reduce((p, leg) => p * leg.probability, 1) : null;
  return { decimal, american: decimal === null ? null : americanOdds(decimal), probability, correlated };
}
export function formatOdds(price) {
  return price == null ? '—' : `${price > 0 ? '+' : ''}${Math.round(price)}`;
}

export function toggleLeg(legs, leg) {
  const existing = legs.find(item => legKey(item) === legKey(leg));
  const remaining = legs.filter(item => legKey(item) !== legKey(leg));
  return existing?.direction === leg.direction && existing.line === leg.line ? remaining : [...remaining, leg];
}
