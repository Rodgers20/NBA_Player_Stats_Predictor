const MIN_STREAK = 5;       // a run shorter than this is not a story
const NEAR_MIN_GAMES = 8;   // fewer games than this cannot support a hit-rate claim
const NEAR_RATE = 0.7;
const NEAR_WINDOW = 20;

const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

/** Mean and median of the numbers present; null when there are none. */
export function summarize(values) {
  const numbers = values.filter(value => value != null && Number.isFinite(value)).sort((a, b) => a - b);
  if (!numbers.length) return null;
  const middle = Math.floor(numbers.length / 2);
  return { mean: mean(numbers), median: numbers.length % 2 ? numbers[middle] : (numbers[middle - 1] + numbers[middle]) / 2 };
}

/** The active game filters in plain words, with a leading space; '' when none are on. */
export function scopeLabel(filters) {
  const parts = [
    filters.homeOnly && 'at home',
    filters.withPlayer && `with ${filters.withPlayer}`,
    filters.withoutPlayer && `without ${filters.withoutPlayer}`,
    filters.result === 'W' && 'in wins',
    filters.result === 'L' && 'in losses',
    filters.rest === '0' && 'on a back-to-back',
  ].filter(Boolean);
  return parts.length ? ` ${parts.join(' ')}` : '';
}

/**
 * A one-line streak or hit-rate sentence for the line, or null when nothing stands out. `games` run oldest to newest.
 * A run counts only while every game is over the line (a push ends it), and needs five games. Without a run, 70% or
 * better over the last 20 games (at least 8) reads as "N of the last M". Written without pronouns on purpose.
 */
export function streakInsight({ player, stat, line, games, scope = '', side = 'Over' }) {
  const facts = streakFacts(games, line, side);
  if (!facts) return null;
  const verb = side === 'Under' ? 'stayed under' : 'exceeded';
  const run = facts.kind === 'straight' ? `${facts.count} straight games` : `${facts.count} of the last ${facts.of} games`;
  return `${player} has ${verb} ${line} ${stat} in ${run}${scope} (${facts.average.toFixed(1)} ${stat}/game average).`;
}

/** The numbers behind a streak sentence: a straight run, or hits out of the last games; null when nothing stands out. */
function streakFacts(games, line, side) {
  const results = games.map(game => game.value).filter(value => value != null);
  if (results.length < MIN_STREAK) return null;
  const clears = value => (side === 'Under' ? value < line : value > line);
  let streak = 0;
  while (streak < results.length && clears(results[results.length - 1 - streak])) streak += 1;
  if (streak >= MIN_STREAK) return { kind: 'straight', count: streak, of: streak, average: mean(results.slice(-streak)) };
  const recent = results.slice(-NEAR_WINDOW);
  const hits = recent.filter(clears).length;
  if (recent.length >= NEAR_MIN_GAMES && hits / recent.length >= NEAR_RATE) return { kind: 'of', count: hits, of: recent.length, average: mean(recent) };
  return null;
}

const INSIGHT_MIN_PRICE = -500; // a heavier favorite than this is not a story worth telling
const INSIGHT_LIMIT = 12;

/**
 * Streak sentences for a game's priced picks, best first: straight runs (longest first), then near-misses by hit rate.
 * Unpriced research, prices worse than -500 and picks without five games of results are skipped. Input is not mutated.
 */
export function gameInsights(props, { limit = INSIGHT_LIMIT } = {}) {
  return props
    .filter(prop => (prop.pick_type === 'pick' || prop.pick_type === 'lean') && prop.line != null && prop.price != null && prop.price >= INSIGHT_MIN_PRICE)
    .flatMap(prop => {
      const side = /^under$/i.test(prop.direction) ? 'Under' : 'Over';
      const games = [...(prop.l10_values ?? [])].reverse().map(value => ({ value })); // the API sends newest first
      const facts = streakFacts(games, prop.line, side);
      if (!facts) return [];
      const stat = prop.stat_label || prop.stat;
      const rank = facts.kind === 'straight' ? 100 + facts.count : (facts.count / facts.of) * 100;
      return [{ prop, rank, text: streakInsight({ player: prop.player, stat, line: prop.line, games, side }) }];
    })
    .sort((a, b) => b.rank - a.rank)
    .slice(0, limit)
    .map(({ prop, text }) => ({ prop, text }));
}
