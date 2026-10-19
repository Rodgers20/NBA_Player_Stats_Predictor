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

/** Plain text a person or a Discord bet bot can read, one leg per line. */
export function slipText(legs, mode, stake) {
  const lines = legs.map(leg => `${leg.player} ${leg.direction} ${leg.line} ${leg.stat} (${formatOdds(leg.price)}) · ${leg.league.toUpperCase()} ${leg.matchup} ${leg.date}`);
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
