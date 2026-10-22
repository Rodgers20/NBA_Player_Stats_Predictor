/** The size of a result above which "margin" is open ended. */
export const MARGIN_CAP = 20;

export const DEFAULT_GAME_FILTERS = Object.freeze({
  homeOnly: false,
  rest: 'any',            // 'any' | '0' | '1' | '2' | '3+' days of rest
  withPlayer: '',         // a teammate who played
  withoutPlayer: '',      // a teammate who sat out
  margin: [0, MARGIN_CAP], // size of the result either way
  result: 'any',          // 'any' | 'W' | 'L'
  minutes: null,          // [low, high] or null for any
});

const marginOn = ([low, high]) => low > 0 || high < MARGIN_CAP;

/** How many filters are narrowing the games. */
export function activeFilterCount(filters) {
  return [
    filters.homeOnly,
    filters.rest !== 'any',
    Boolean(filters.withPlayer),
    Boolean(filters.withoutPlayer),
    marginOn(filters.margin ?? DEFAULT_GAME_FILTERS.margin),
    filters.result !== 'any',
    filters.minutes != null,
  ].filter(Boolean).length;
}

const restMatches = (rest, wanted) => rest != null && (wanted === '3+' ? rest >= 3 : rest === Number(wanted));

/**
 * The games that satisfy every active filter. A filter on something a game has no data for (rest, margin,
 * result, minutes) leaves that game out rather than guessing. "With" a teammate means he did not sit out.
 */
export function applyGameFilters(games, filters = DEFAULT_GAME_FILTERS) {
  const [low, high] = filters.margin ?? DEFAULT_GAME_FILTERS.margin;
  const limited = marginOn([low, high]);
  return games.filter(game => {
    if (filters.homeOnly && game.is_home !== true) return false;
    if (filters.rest !== 'any' && !restMatches(game.rest, filters.rest)) return false;
    const out = game.out ?? [];
    if (filters.withPlayer && out.includes(filters.withPlayer)) return false;
    if (filters.withoutPlayer && !out.includes(filters.withoutPlayer)) return false;
    if (limited) {
      if (game.margin == null) return false;
      const size = Math.abs(game.margin);
      if (size < low || (high < MARGIN_CAP && size > high)) return false;
    }
    if (filters.result !== 'any' && game.won !== (filters.result === 'W')) return false;
    if (filters.minutes && (game.minutes == null || game.minutes < filters.minutes[0] || game.minutes > filters.minutes[1])) return false;
    return true;
  });
}

/** Teammates who actually sat out at least one of these games, most often first: the only ones a with/without split can tell apart. */
export function teammateOptions(games) {
  const counts = new Map();
  for (const game of games) for (const name of game.out ?? []) counts.set(name, (counts.get(name) ?? 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([name]) => name);
}

/** Counts of values in `bins` even bins from `low` to `high`; values outside the range land in the edge bins. */
export function histogram(values, low, high, bins) {
  const counts = Array.from({ length: bins }, () => 0);
  const width = (high - low) / bins || 1;
  for (const value of values) {
    if (value == null || !Number.isFinite(value)) continue;
    counts[Math.min(bins - 1, Math.max(0, Math.floor((value - low) / width)))] += 1;
  }
  return counts;
}

/** Whole-minute bounds of the minutes the games show; a full game when there are none. */
export function minutesRange(games) {
  const minutes = games.map(game => game.minutes).filter(value => value != null && Number.isFinite(value));
  return minutes.length ? [Math.floor(Math.min(...minutes)), Math.ceil(Math.max(...minutes))] : [0, 48];
}
