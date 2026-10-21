import { decimalOdds } from './bet-slip.mjs';

/**
 * A slip leg for a Trends card, or null when it cannot be priced honestly.
 * A live sportsbook line brings its own price. A derived line has none, so the price must come from
 * the user (`price`); it is never guessed. An entered price wins over the live one.
 */
export function trendLeg({ card, league, date, price }) {
  const line = card.live?.line ?? card.line;
  const chosen = price ?? card.live?.price;
  if (!Number.isFinite(line) || decimalOdds(chosen) === null) return null;
  return {
    league, date, player: card.player, stat: card.stat,
    direction: card.side === 'under' ? 'Under' : 'Over',
    line, price: chosen, probability: null,
    matchup: card.opponent ? `${card.team} vs ${card.opponent}` : card.team,
    team: card.team,
    addedAt: new Date().toISOString(),
  };
}
