export function playerKey(name) {
  return Array.from(new TextEncoder().encode(name), b => b.toString(16).padStart(2, '0')).join('');
}

export function filterProps(data, params = {}) {
  const eligible = data.props.filter(p =>
    (!params.direction || params.direction === 'all' || p.direction.toLowerCase() === String(params.direction).toLowerCase()) &&
    (!params.game || p.game_matchup.toLowerCase().includes(String(params.game).toLowerCase())) &&
    (!params.location || params.location === 'all' || p.is_home_today === (params.location === 'home')));
  const stat_counts = {};
  for (const prop of eligible) stat_counts[prop.stat] = (stat_counts[prop.stat] ?? 0) + 1;
  let props = eligible.filter(p =>
    (!params.stat || (params.stat === 'COMBO' ? p.is_combo : p.stat === params.stat)) &&
    (!params.locks_only || p.is_lock) && (!params.combos_only || p.is_combo));
  const key = params.sort === 'hit_rate' ? 'hit_rate' : 'ev';
  props.sort((a, b) => Number(b.is_lock) - Number(a.is_lock) || (b[key] ?? -Infinity) - (a[key] ?? -Infinity));
  return { ...data, stat_counts, count: props.length, props: props.slice(0, Number(params.limit ?? 100)) };
}

export function windowChart(data, count) {
  const games = data.games.slice(-count);
  const mean = rows => {
    const values = rows.map(r => r.value).filter(v => v !== null);
    return values.length ? Math.round(values.reduce((a, b) => a + b, 0) / values.length * 10) / 10 : null;
  };
  return { ...data, games, avg: mean(games), l5_avg: mean(games.slice(-5)) };
}

/** Rebuild the API chart response from the compact, per-player history export. */
export function seriesChart(player, series, stat, count = 20) {
  const columns = stat.toUpperCase().split('+');
  const rows = series.games.slice(-count).map(row => {
    const values = columns.map(column => row[column.toLowerCase()]);
    const value = values.every(item => typeof item === 'number' && Number.isFinite(item))
      ? values.reduce((sum, item) => sum + item, 0) : null;
    return { date: row.date, opponent: row.opponent, season: row.season,
      is_home: row.is_home, value, hit: null };
  });
  const mean = values => {
    const numbers = values.map(row => row.value).filter(value => value !== null);
    return numbers.length ? Math.round(numbers.reduce((a, b) => a + b, 0) / numbers.length * 10) / 10 : null;
  };
  return { player, stat: stat.toUpperCase(), line: null, direction: 'over',
    avg: mean(rows), l5_avg: mean(rows.slice(-5)), games: rows };
}
