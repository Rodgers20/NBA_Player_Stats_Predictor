/** Pure helpers for the defense-vs-position heatmap. Rank 1 = softest defense (allows the most). */
export function pivotDefense(rankings, stat) {
  const rows = new Map();
  const positions = new Set();
  for (const item of rankings.filter(r => r.stat === stat)) {
    positions.add(item.position);
    const row = rows.get(item.team) ?? { team: item.team, cells: {} };
    rows.set(item.team, { ...row, cells: { ...row.cells, [item.position]: item } });
  }
  const list = [...rows.values()].map(row => {
    const ranks = Object.values(row.cells).map(cell => cell.rank);
    return { ...row, avgRank: ranks.length ? ranks.reduce((a, b) => a + b, 0) / ranks.length : null };
  });
  return { positions: [...positions].sort(), rows: list };
}

/** key: 'team' | 'avg' | a position code. Missing cells sort last; returns a new array. */
export function sortRows(rows, key, dir = 'asc') {
  const sign = dir === 'desc' ? -1 : 1;
  const value = row => key === 'team' ? row.team : key === 'avg' ? row.avgRank : row.cells[key]?.rank ?? null;
  return [...rows].sort((a, b) => {
    const x = value(a), y = value(b);
    if (x == null || y == null) return (x == null) - (y == null) || a.team.localeCompare(b.team);
    return (typeof x === 'string' ? x.localeCompare(y) : x - y) * sign || a.team.localeCompare(b.team);
  });
}

/** 0 (toughest) to 1 (softest) for colouring. */
export const softness = (rank, of) => of > 1 ? 1 - (rank - 1) / (of - 1) : 0.5;
