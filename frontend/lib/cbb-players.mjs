// Ranked college player board: which stats exist, how a row is valued, filtered and ordered.

/** Columns in display order. `low` marks stats where a smaller number is better (turnovers). */
export const STATS = [
  { key: 'pts', label: 'PTS', title: 'Points per game', digits: 1 },
  { key: 'reb', label: 'REB', title: 'Rebounds per game', digits: 1 },
  { key: 'ast', label: 'AST', title: 'Assists per game', digits: 1 },
  { key: 'min', label: 'MIN', title: 'Minutes per game', digits: 1 },
  { key: 'stl', label: 'STL', title: 'Steals per game', digits: 1 },
  { key: 'blk', label: 'BLK', title: 'Blocks per game', digits: 1 },
  { key: 'tpm', label: '3PM', title: 'Three-pointers made per game', digits: 1 },
  { key: 'tov', label: 'TO', title: 'Turnovers per game (fewer is better)', digits: 1, low: true },
  { key: 'fg', label: 'FG%', title: 'Field goal percentage', digits: 1 },
  { key: 'tp', label: '3P%', title: 'Three-point percentage', digits: 1 },
  { key: 'ft', label: 'FT%', title: 'Free throw percentage', digits: 1 },
  { key: 'pra', label: 'PRA', title: 'Points + rebounds + assists per game', digits: 1 },
  { key: 'stocks', label: 'STK', title: 'Steals + blocks per game', digits: 1 },
  { key: 'gp', label: 'GP', title: 'Games played', digits: 0 },
];

export const POSITIONS = [
  { value: '', label: 'All positions' },
  { value: 'G', label: 'Guards' },
  { value: 'F', label: 'Forwards' },
  { value: 'C', label: 'Centers' },
];

const num = v => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** One stat for one season's line, including the two derived ones. Null when the player has no such line. */
export function statValue(line, key) {
  if (!line) return null;
  if (key === 'pra') return [line.pts, line.reb, line.ast].every(v => num(v) != null) ? line.pts + line.reb + line.ast : null;
  if (key === 'stocks') return [line.stl, line.blk].every(v => num(v) != null) ? line.stl + line.blk : null;
  return num(line[key]);
}

export function formatStat(value, key) {
  if (value == null) return '–';
  const spec = STATS.find(s => s.key === key);
  return value.toFixed(spec ? spec.digits : 1);
}

/** 2027 -> "2026-27". */
export function seasonLabel(season) {
  return `${season - 1}-${String(season).slice(-2)}`;
}

/** Start on the running season once enough players have logged real games; until then show last season's finished numbers. */
export function defaultSeason(payload, minPlayers = 150, minGames = 5) {
  const current = String(payload.season);
  const settled = payload.players.filter(p => (p.stats[current]?.gp ?? 0) >= minGames).length;
  return settled >= minPlayers ? payload.season : payload.seasons[1] ?? payload.season;
}

const norm = s => (s ?? '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9 ]/g, '');

/** Filter then order. Each row gets `line`, the sorted `value` and a 1-based `rank` within the filtered, sorted list. */
export function boardRows(players, { season, sort = 'pts', dir = -1, minGames = 1, query = '', position = '' } = {}) {
  const needle = norm(query).trim();
  const rows = [];
  for (const p of players) {
    const line = p.stats[String(season)];
    if (!line || (line.gp ?? 0) < minGames) continue;
    if (position && !(p.pos ?? '').includes(position)) continue;
    if (needle && !norm(`${p.name} ${p.team}`).includes(needle)) continue;
    const value = statValue(line, sort);
    if (value == null) continue;
    rows.push({ ...p, line, value });
  }
  rows.sort((a, b) => (a.value - b.value) * dir || a.name.localeCompare(b.name));
  return rows.map((r, i) => ({ ...r, rank: i + 1 }));
}

/** Default direction when a column header is first picked: best first. */
export function bestFirst(key) {
  return STATS.find(s => s.key === key)?.low ? 1 : -1;
}

export function headshot(id) {
  return `https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/${id}.png`;
}
