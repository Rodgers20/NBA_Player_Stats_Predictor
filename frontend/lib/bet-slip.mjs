import { suggestedStake, sanitizeBankroll } from './bankroll.mjs';
import { legText } from './team-legs.mjs';

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

export const MAX_LEGS = 15;

/** What a winning bet returns, stake included; null without a valid stake or price. */
export function payout(stake, price) {
  const decimal = decimalOdds(price);
  return decimal === null || !(stake > 0) ? null : stake * decimal;
}

/** Singles: each leg is its own bet at `stake`. Returns null payout unless every leg is priced. */
export function summarizeSingles(legs, stake) {
  const returns = legs.map(leg => payout(stake, leg.price));
  const complete = stake > 0 && returns.length > 0 && returns.every(r => r !== null);
  return { bets: legs.length, totalStake: stake > 0 ? stake * legs.length : null, payout: complete ? returns.reduce((a, b) => a + b, 0) : null };
}

/**
 * Fractional-Kelly stakes in dollars: one per leg (as a single bet) and a total for the slip.
 * A parlay is staked as one bet on its combined price, using the correlation-adjusted chance when known.
 */
export function slipStakes(legs, mode, bankroll, correlatedProb = null) {
  const perLeg = legs.map(leg => suggestedStake(leg.probability, leg.price, bankroll));
  if (mode === 'singles' || legs.length < 2) return { perLeg, total: Math.round(perLeg.reduce((a, b) => a + b, 0) * 100) / 100 };
  const s = summarizeSlip(legs);
  const p = Number.isFinite(correlatedProb) ? correlatedProb : s.probability;
  return { perLeg, total: suggestedStake(p, s.american, bankroll) };
}

/** Untrusted persisted slip (older shapes have no bankroll or team): keep what is valid, default the rest. */
export function sanitizePersistedSlip(value) {
  const legs = Array.isArray(value?.legs)
    ? value.legs.filter(l => l && typeof l.player === 'string' && typeof l.stat === 'string' && typeof l.date === 'string' && typeof l.league === 'string' && typeof l.matchup === 'string' && Number.isFinite(l.line) && (l.price == null || Number.isFinite(l.price))).slice(0, MAX_LEGS)
    : [];
  return { legs, mode: value?.mode === 'singles' ? 'singles' : 'parlay',
    stake: Number.isFinite(value?.stake) && value.stake > 0 ? value.stake : null, bankroll: sanitizeBankroll(value?.bankroll) };
}

/** Plain text a person or a Discord bet bot can read, one leg per line. */
export function slipText(legs, mode, stake) {
  const lines = legs.map(leg => `${legText(leg)} · ${leg.league.toUpperCase()} ${leg.matchup} ${leg.date}`);
  if (mode === 'singles' && legs.length > 1) return [`${legs.length} singles${stake ? `, $${stake} each` : ''}`, ...lines].join('\n');
  const american = summarizeSlip(legs).american;
  const head = legs.length === 1 ? 'Single' : `${legs.length}-leg parlay`;
  return [`${head}${american !== null ? ` ${formatOdds(american)}` : ''}${stake ? `, $${stake}` : ''}`, ...lines].join('\n');
}

/* ---- shareable link: the legs travel in the URL, so a slip needs no account or server ---- */

const toBase64Url = s => {
  let bin = '';
  for (const b of new TextEncoder().encode(s)) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const fromBase64Url = s => new TextDecoder().decode(Uint8Array.from(atob(s.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0)));

export function encodeSlip(legs, mode, stake) {
  const packed = legs.map(l => [l.league, l.date, l.player, l.stat, l.direction, l.line, l.price, l.probability ?? null, l.matchup]);
  return toBase64Url(JSON.stringify({ v: 1, m: mode, s: stake ?? null, l: packed }));
}

/** Untrusted input: anything malformed returns null or drops the bad leg. */
export function decodeSlip(code) {
  if (!code) return null;
  try {
    const raw = JSON.parse(fromBase64Url(code));
    if (raw?.v !== 1 || !Array.isArray(raw.l)) return null;
    const legs = raw.l.slice(0, MAX_LEGS).flatMap(p => {
      if (!Array.isArray(p)) return [];
      const [league, date, player, stat, direction, line, price, probability, matchup] = p;
      if (!['nba', 'wnba'].includes(league) || ![date, player, stat, direction, matchup].every(v => typeof v === 'string')) return [];
      if (typeof line !== 'number' || decimalOdds(price) === null) return [];
      return [{ league, date, player, stat, direction, line, price, probability: typeof probability === 'number' ? probability : null, matchup, addedAt: '' }];
    });
    const stake = typeof raw.s === 'number' && raw.s > 0 ? raw.s : null;
    return { legs, mode: raw.m === 'singles' ? 'singles' : 'parlay', stake };
  } catch {
    return null;
  }
}
