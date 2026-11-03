/** Board filtering and sorting for the NHL props page. Pure so it can be tested without React. */

const DEFAULT_LINE = { goal: 0.5, sog: 2.5, assists: 0.5, points: 0.5 };

export const SORTS = [
  { key: 'p', label: 'Probability' },
  { key: 'proj', label: 'Projection' },
  { key: 'l10', label: 'Last 10 hit rate' },
];

/** Sorted unique lines any player has for a market. */
export function lineOptions(players, market) {
  const set = new Set();
  for (const p of players) for (const l of p.markets?.[market]?.lines ?? []) set.add(l.line);
  return [...set].sort((a, b) => a - b);
}

/** The line to show first: the market's usual one when posted, else the lowest available. */
export function initialLine(options, market) {
  return options.includes(DEFAULT_LINE[market]) ? DEFAULT_LINE[market] : options[0] ?? null;
}

const fold = s => String(s ?? '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();

/**
 * One row per player that has the chosen line in the market. Filters: `date`, `gameId`, `team`
 * (player's team or opponent), `query` (name or team). Sort keys: p, proj, l10; ties fall back to name.
 */
export function boardRows(players, { market, line, date, gameId, team, query = '', sort = 'p' }) {
  const q = fold(query).trim();
  const rows = [];
  for (const player of players) {
    if (date && player.date !== date) continue;
    if (gameId && player.game_id !== Number(gameId)) continue;
    if (team && player.team !== team && player.opp !== team) continue;
    if (q && !fold(player.name).includes(q) && !fold(player.team).includes(q)) continue;
    const m = player.markets?.[market];
    const l = m?.lines.find(x => x.line === line);
    if (!m || !l) continue;
    rows.push({ player, market: m, line: l });
  }
  const key = { p: r => r.line.p_over, proj: r => r.market.projection, l10: r => r.line.hit_l10 ?? -1 }[sort] ?? (r => r.line.p_over);
  return rows.sort((a, b) => key(b) - key(a) || a.player.name.localeCompare(b.player.name));
}

/** Distinct games on the board for the game filter, in start order. */
export function boardGames(players, date) {
  const seen = new Map();
  for (const p of players) {
    if (date && p.date !== date) continue;
    if (!seen.has(p.game_id)) seen.set(p.game_id, { id: p.game_id, start: p.start, away: p.home ? p.opp : p.team, home: p.home ? p.team : p.opp });
  }
  return [...seen.values()].sort((a, b) => a.start.localeCompare(b.start));
}
