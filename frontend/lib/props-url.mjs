const STATS = new Set(['', 'PTS', 'AST', 'REB', 'FG3M', 'BLK', 'STL', 'PTS+AST', 'PTS+REB', 'AST+REB', 'PTS+AST+REB', 'DD', 'TD', 'ML', 'SPREAD', 'TOTAL']);
const DIRECTIONS = new Set(['All', 'Over', 'Under']);
const LOCATIONS = new Set(['All', 'Home', 'Away']);
const SORTS = new Set(['hit_rate', 'ev', 'edge']);
const KEYS = ['stat', 'direction', 'location', 'sort', 'game', 'minStars', 'q'];

/** Printable, trimmed, length-capped text; control characters never survive. */
const clean = (text, max) => text.replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, max);
const pick = (value, allowed, fallback) => (value !== null && allowed.has(value) ? value : fallback);

/**
 * Board filters from a URL query. Every value is whitelisted or capped, so a hand-edited
 * or hostile URL falls back to the defaults and never reaches the API.
 */
export function parseFilters(search, defaults) {
  const params = new URLSearchParams(search);
  const stat = params.get('stat');
  const stars = params.get('minStars');
  return {
    ...defaults,
    stat: pick(stat === null ? null : stat.replace(/ /g, '+'), STATS, defaults.stat), // a typed "+" arrives as a space
    direction: pick(params.get('direction'), DIRECTIONS, defaults.direction),
    location: pick(params.get('location'), LOCATIONS, defaults.location),
    sort: pick(params.get('sort'), SORTS, defaults.sort),
    game: clean(params.get('game') ?? defaults.game, 60),
    minStars: stars !== null && /^[0-5]$/.test(stars) ? Number(stars) : defaults.minStars,
    q: clean(params.get('q') ?? '', 40),
  };
}

/** Only values that differ from the defaults, in a stable order, without a leading "?". */
export function filtersToSearch(filters, defaults) {
  const params = new URLSearchParams();
  for (const key of KEYS) {
    const value = filters[key];
    const base = key === 'q' ? '' : defaults[key];
    // An empty value is written too when the default is not empty, so "All" stays selectable.
    if (value !== undefined && value !== base) params.set(key, String(value));
  }
  return params.toString();
}

/** A game filter that the known slate does not contain can never match; an unknown or empty slate proves nothing. */
export function isStaleGame(game, matchups) {
  return Boolean(game) && Array.isArray(matchups) && matchups.length > 0 && !matchups.includes(game);
}
