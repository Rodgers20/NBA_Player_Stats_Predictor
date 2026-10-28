import { hitSummary } from './hit-rate.mjs';

export const WINDOW_KEYS = [5, 10, 20, 'h2h', 'home', 'away', 'current', 'previous'];

/** The games a window covers, oldest first. `ctx` carries the next opponent and the season labels. */
export function selectWindow(games, period, ctx = {}) {
  const lastWord = game => game.opponent.split(' ').at(-1);
  switch (period) {
    case 'h2h': return ctx.nextOpponent ? games.filter(game => lastWord(game) === ctx.nextOpponent).slice(-10) : [];
    case 'home': return games.filter(game => game.is_home === true || (game.is_home == null && game.opponent.includes('vs.'))).slice(-10);
    case 'away': return games.filter(game => game.is_home === false || (game.is_home == null && game.opponent.includes('@'))).slice(-10);
    case 'current': return ctx.currentSeason ? games.filter(game => game.season === ctx.currentSeason) : [];
    case 'previous': return ctx.previousSeason ? games.filter(game => game.season === ctx.previousSeason) : [];
    default: return games.slice(-period);
  }
}

/**
 * Hit counts per window at one line, graded over the line like everywhere else: a tie is a push and
 * leaves the denominator. Games without a value are not games. `n` counts real results.
 */
export function windowCounts(games, threshold, ctx = {}, side = 'Over') {
  return WINDOW_KEYS.map(key => {
    const values = selectWindow(games, key, ctx).map(game => game.value).filter(value => value != null);
    const { n, hits, misses, pushes, rate } = hitSummary(values, threshold, side);
    return { key, n, hits, decided: hits + misses, pushes, rate };
  });
}
