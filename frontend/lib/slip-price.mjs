import { isTeamLeg } from './team-legs.mjs';

export const MAX_PRICED_LEGS = 12; // matches api/routes/slip.py

const parts = stat => String(stat || '').toUpperCase().split('+').filter(Boolean);
const norm = name => String(name || '').trim().toLowerCase();

/** Body for POST /api/slip/price, or null when the slip cannot be priced (needs 2+ legs with model probabilities). */
export function buildPriceRequest(legs, teamByPlayer = {}) {
  if (legs.length < 2 || legs.length > MAX_PRICED_LEGS) return null;
  if (legs.some(isTeamLeg)) return null; // the service models player stats only; the slip falls back to independent maths
  if (!legs.every(leg => Number.isFinite(leg.probability) && leg.probability > 0 && leg.probability < 1)) return null;
  return { legs: legs.map(leg => ({
    player: leg.player, team: leg.team || teamByPlayer[norm(leg.player)] || '',
    game: leg.matchup ? `${leg.league}:${leg.date}:${leg.matchup}` : '',
    stat: leg.stat, side: leg.direction === 'Under' ? 'Under' : 'Over', prob: leg.probability,
    ...(Number.isFinite(leg.price) && Math.abs(leg.price) >= 100 ? { price: leg.price } : {}),
  })) };
}

/** One short sentence per correlated leg, e.g. "Correlated with leg 2 (+0.60): same player, one prop is part of the other". */
export function describeCorrelation(item) {
  if (!item?.notes?.length) return null;
  return item.notes.map(note => `Correlated with leg ${note.with} (${note.rho > 0 ? '+' : ''}${note.rho.toFixed(2)}): ${note.reason}`).join('. ');
}

/**
 * Other props for the same player that share a stat with this leg, so they move together.
 * Only mechanical same-player overlap is suggested; cross-player effects are too weak to call "strong".
 */
export function relatedProps(leg, props, slipLegs, limit = 2) {
  const taken = new Set(slipLegs.map(item => `${norm(item.player)}|${item.stat}`));
  const legParts = parts(leg.stat);
  return props.filter(prop => norm(prop.player) === norm(leg.player) && prop.stat !== leg.stat &&
      Number.isFinite(prop.line) && Number.isFinite(prop.price) &&
      parts(prop.stat).some(part => legParts.includes(part)) && !taken.has(`${norm(prop.player)}|${prop.stat}`))
    .slice(0, limit);
}

export async function fetchSlipPrice(baseUrl, body, signal) {
  const response = await fetch(`${baseUrl}/api/slip/price`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
  if (!response.ok) throw new Error(`Slip pricing unavailable (${response.status})`);
  return response.json();
}
