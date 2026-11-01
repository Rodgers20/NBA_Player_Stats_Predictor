/** Pure helpers for the college basketball pages. Spreads are betting-style from the HOME team. */

/** "-4.5" / "+4.5" / "PK" for one side of the game. */
export function spreadText(spreadHome, side) {
  if (spreadHome == null) return null;
  const value = side === 'home' ? spreadHome : -spreadHome;
  return value === 0 ? 'PK' : value > 0 ? `+${value}` : `${value}`;
}

export function american(odds) {
  if (odds == null) return null;
  return odds > 0 ? `+${Math.round(odds)}` : `${Math.round(odds)}`;
}

export const pct = (value, digits = 0) => (value == null ? '—' : `${(value * 100).toFixed(digits)}%`);
export const signed = (value, digits = 1) => (value == null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(digits)}`);

/** The side the vig-free moneyline favours, or null without a market price. */
export function marketSide(pick) {
  if (!pick || pick.market_p_home == null || pick.market_p_home === 0.5) return null;
  return pick.market_p_home > 0.5 ? 'home' : 'away';
}

/** True when the model picks a different winner than the sportsbook favourite. */
export function disagrees(game) {
  const market = marketSide(game.pick);
  return market != null && game.pick.winner !== market;
}

/** Points between the model's margin and the book's, home-minus-away. Null without both. */
export function spreadEdge(game) {
  if (!game.pick || game.line?.spread_home == null) return null;
  return Math.round((game.pick.proj_margin + game.line.spread_home) * 10) / 10;
}

/** The day to open: today when it has games, else the closest day that does (later days win ties). */
export function defaultDay(days, today) {
  const withGames = days.filter(day => day.games > 0);
  if (!withGames.length) return days.find(day => day.date === today)?.date ?? days[0]?.date ?? today;
  const ms = date => Date.parse(`${date}T12:00:00Z`);
  const target = ms(today);
  return withGames.reduce((best, day) => {
    const gap = Math.abs(ms(day.date) - target), bestGap = Math.abs(ms(best.date) - target);
    return gap < bestGap || (gap === bestGap && ms(day.date) > ms(best.date)) ? day : best;
  }).date;
}

/** Wins and losses of the pre-tip calls on graded games (pushes and ungraded calls are skipped). */
export function dayRecord(games) {
  const rec = () => ({ wins: 0, losses: 0 });
  const out = { winner: rec(), ats: rec(), total: rec(), graded: 0 };
  for (const game of games) {
    if (!game.result) continue;
    out.graded += 1;
    for (const [key, ok] of [['winner', game.result.winner_correct], ['ats', game.result.ats_correct], ['total', game.result.total_correct]]) {
      if (ok === true) out[key].wins += 1;
      else if (ok === false) out[key].losses += 1;
    }
  }
  return out;
}

/** Filter by 'all' | 'top25' | 'lines' | 'model' and an optional conference name. */
export function filterGames(games, { view = 'all', conference = '' } = {}) {
  return games.filter(game => {
    if (view === 'top25' && game.home.rank == null && game.away.rank == null) return false;
    if (view === 'lines' && !game.line) return false;
    if (view === 'disagree' && !disagrees(game)) return false;
    if (conference && game.home.conference !== conference && game.away.conference !== conference) return false;
    return true;
  });
}

/** Sorted conference names appearing on a slate. */
export function conferencesOf(games) {
  const names = new Set();
  for (const game of games) for (const team of [game.home, game.away]) if (team.conference) names.add(team.conference);
  return [...names].sort((a, b) => a.localeCompare(b));
}

/** "6:00 PM ET" from a 24h Eastern "HH:MM". */
export function tipText(time) {
  const [h, m] = String(time ?? '').split(':').map(Number);
  if (!Number.isFinite(h) || !Number.isFinite(m)) return 'TBA';
  return `${h % 12 || 12}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'} ET`;
}

export const CONFIDENCE_LABEL = { high: 'High confidence', medium: 'Medium confidence', low: 'Low confidence', coin_flip: 'Coin flip' };

/** A team colour readable on the dark card: the primary unless it is too dark, then the alternate, else a neutral light teal. */
export function barColor(team) {
  const lum = hex => {
    const clean = String(hex ?? '').replace('#', '');
    if (!/^[a-f0-9]{6}$/i.test(clean)) return 0;
    const [r, g, b] = [0, 2, 4].map(i => parseInt(clean.slice(i, i + 2), 16) / 255);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  if (lum(team.color) >= 0.2) return `#${team.color.replace('#', '')}`;
  if (lum(team.alt_color) >= 0.2) return `#${team.alt_color.replace('#', '')}`;
  return '#7aa7b0';
}

/** Hit/miss result of one graded call as 'hit' | 'miss' | 'push' | null. */
export function mark(correct, result) {
  if (correct === true) return 'hit';
  if (correct === false) return 'miss';
  return result === 'push' ? 'push' : null;
}
