/** Pure helpers behind the prop panel chart: window slicing and hit/miss grading against a line. */
export const WINDOW_KEYS = ['L5', 'L10', 'L20', 'H2H', 'Season'];
const SIZES = { L5: 5, L10: 10, L20: 20 };
const opponentOf = game => String(game.opponent ?? '').trim().split(/\s+/).at(-1);

/** `games` is oldest to newest. H2H needs the opponent abbreviation; Season is the latest season present. */
export function windowGames(games, key, opponent = '') {
  if (key in SIZES) return games.slice(-SIZES[key]);
  if (key === 'H2H') return opponent ? games.filter(game => opponentOf(game) === opponent) : [];
  if (key === 'Season') {
    const latest = [...games].reverse().find(game => game.season)?.season;
    return latest ? games.filter(game => game.season === latest) : games;
  }
  return games;
}

/** Adds `hit` per game; a game exactly on the line is a push (null), as on the server. */
export function gradeGames(games, line, direction = 'over') {
  const under = String(direction).toLowerCase() === 'under';
  return games.map(game => ({
    ...game,
    hit: line == null || game.value == null || game.value === line ? null : under ? game.value < line : game.value > line,
  }));
}

export function hitSummary(graded) {
  const decided = graded.filter(game => game.hit !== null);
  return { hits: decided.filter(game => game.hit).length, total: decided.length };
}

export function average(games) {
  const values = games.map(game => game.value).filter(value => value != null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}
