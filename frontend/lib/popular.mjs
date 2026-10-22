import { windowCell } from './prop-rows.mjs';

const MIN_GAMES = 5;       // a hit rate over fewer games is noise
const MIN_PRICE = -500;    // heavier favorites than this are not worth showing as "popular" plays
const DEFAULT_LIMIT = 25;

/** Whether a game is under way or over; unknown or missing data is never treated as started. */
export function isStarted(game, now = Date.now()) {
  if (!game) return false;
  if (game.status_state === 'in' || game.status_state === 'post') return true;
  const start = game.start_time ? Date.parse(game.start_time) : NaN;
  return !Number.isNaN(start) && start <= now;
}

const eligible = (prop, startedGames) =>
  (prop.pick_type === 'pick' || prop.pick_type === 'lean')
  && prop.line != null && prop.price != null && prop.price >= MIN_PRICE
  && !startedGames.has(prop.game_matchup);

const byEv = (a, b) => (b.ev ?? -Infinity) - (a.ev ?? -Infinity);

/**
 * Today's most compelling priced plays. "hit" ranks by last-10 hit rate (edge, then sample size break ties);
 * "edge" ranks real picks by expected value. Games already under way are dropped; a game the slate does not
 * list is kept, because missing data proves nothing. The input is not mutated.
 */
export function popularBoard(props, games, { by = 'hit', now = Date.now(), limit = DEFAULT_LIMIT } = {}) {
  const startedGames = new Set((games ?? []).filter(game => isStarted(game, now)).map(game => game.matchup));
  const rows = props.filter(prop => eligible(prop, startedGames));
  if (by === 'edge') return rows.filter(prop => prop.pick_type === 'pick').sort(byEv).slice(0, limit);
  return rows
    .map(prop => ({ prop, window: windowCell(prop, 'L10') }))
    .filter(({ window }) => window && window.total >= MIN_GAMES)
    .sort((a, b) => b.window.pct - a.window.pct || byEv(a.prop, b.prop) || b.window.total - a.window.total)
    .slice(0, limit)
    .map(({ prop }) => prop);
}
