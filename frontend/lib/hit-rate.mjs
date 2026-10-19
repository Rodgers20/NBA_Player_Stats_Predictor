/** Whether one result clears the line on the given side. A tie is a push (the stake comes back). */
export function gradeSide(value, line, side) {
  if (value === line) return 'push';
  return (side === 'Under' ? value < line : value > line) ? 'hit' : 'miss';
}

/**
 * How often a side cleared the line. Pushes are left out of the rate because a book refunds them.
 * `recent` is the last `lastN` results, oldest first.
 */
export function hitSummary(values, line, side = 'Over', lastN = 5) {
  const results = values.map(value => ({ value, result: gradeSide(value, line, side) }));
  const count = kind => results.filter(item => item.result === kind).length;
  const hits = count('hit'), misses = count('miss'), pushes = count('push');
  const decided = hits + misses;
  return { n: values.length, hits, misses, pushes,
    rate: decided ? Math.round((hits / decided) * 100) : null,
    recent: results.slice(-lastN) };
}

/** Grades a journal bet from the player's game on its date; null while that game has no result. */
export function gradeBet(bet, games) {
  const game = games.find(item => item.date === bet.game_date && item.value != null);
  if (!game) return null;
  return { hit: 'win', miss: 'loss', push: 'push' }[gradeSide(game.value, bet.line, bet.side)];
}
