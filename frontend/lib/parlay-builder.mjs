import { decimalOdds } from './bet-slip.mjs';

/** How each risk level ranks a pick: chance it hits, edge over the book, or expected value per dollar. */
const SCORES = {
  safe: prop => prop.model_prob,
  edge: prop => prop.edge_pct,
  payout: prop => prop.ev,
};
const score = (risk, prop) => {
  const value = (SCORES[risk] ?? SCORES.safe)(prop);
  return Number.isFinite(value) ? value : -Infinity;
};

/** A board row the builder may use: a priced model pick with a real over/under line. Same rule the row's Add button uses. */
export function canBuildFrom(prop) {
  return (prop.pick_type ? prop.pick_type !== 'research' : Boolean(prop.recommendation_eligible ?? prop.has_live_odds))
    && Boolean(prop.has_live_odds) && Number.isFinite(prop.line) && decimalOdds(prop.price) !== null
    && /^(over|under)$/i.test(prop.direction ?? '')
    && Number.isFinite(prop.model_prob) && prop.model_prob > 0 && prop.model_prob < 1;
}

/** The slip leg for a board row. */
export function propLeg(prop, league, date) {
  return { league, date, player: prop.player, stat: prop.stat, direction: prop.direction, line: prop.line, price: prop.price,
    probability: prop.model_prob ?? null, matchup: prop.game_matchup, team: prop.team, addedAt: new Date().toISOString() };
}

/**
 * Picks `legs` props from the model's board. Required players go in first (their best-ranked prop), then the rest by risk.
 * One leg per player, so no leg is part of another; legs spread across games before any game is used twice.
 * Returns fewer legs when the board is thin. The input is not mutated.
 */
export function buildParlay(props, { legs = 3, risk = 'safe', include = [] } = {}) {
  const ranked = props.filter(canBuildFrom).sort((a, b) => score(risk, b) - score(risk, a));
  const picked = [];
  const takes = (prop, spread) => !picked.some(p => p.player === prop.player) && !(spread && picked.some(p => p.game_matchup === prop.game_matchup));
  for (const name of include) {
    const prop = ranked.find(p => p.player === name);
    if (prop && picked.length < legs && takes(prop, false)) picked.push(prop);
  }
  for (const spread of [true, false]) {
    for (const prop of ranked) if (picked.length < legs && takes(prop, spread)) picked.push(prop);
  }
  return picked;
}
