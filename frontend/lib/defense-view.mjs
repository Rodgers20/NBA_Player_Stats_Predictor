/** Pure helpers for showing defense-vs-position context. Rank 1 = softest defense (allows the most). */
export const DEF_STATS = ['PTS', 'REB', 'AST'];
const BANDS = [[0.2, 'A'], [0.4, 'B'], [0.6, 'C'], [0.8, 'D']];

/** Mirrors utils/defense_rank.grade: A (softest) to F (toughest). */
export function gradeFor(rank, of) {
  const fraction = of > 1 ? (rank - 1) / (of - 1) : 0.5;
  return (BANDS.find(([limit]) => fraction <= limit) ?? [1, 'F'])[1];
}

/** 'G-F' -> 'G'; anything outside G/F/C -> null. */
export function basePosition(position) {
  const letter = String(position ?? '').trim().toUpperCase().charAt(0);
  return ['G', 'F', 'C'].includes(letter) ? letter : null;
}

/** Per PTS/REB/AST: the team's rank against one position plus the league mean allowed for that position. */
export function positionMatchup(rankings, team, position) {
  const out = {};
  for (const stat of DEF_STATS) {
    const column = rankings.filter(r => r.stat === stat && r.position === position);
    const cell = column.find(r => r.team === team);
    if (!cell) continue;
    const leagueAvg = column.reduce((sum, r) => sum + r.allowed, 0) / column.length;
    out[stat] = { ...cell, leagueAvg, grade: gradeFor(cell.rank, cell.of) };
  }
  return out;
}

/** Team-wide view: mean rank per stat across positions, plus the softest and toughest single cells. */
export function teamDefense(rankings, team) {
  const cells = rankings.filter(r => r.team === team);
  if (!cells.length) return null;
  const byStat = {};
  for (const stat of DEF_STATS) {
    const mine = cells.filter(r => r.stat === stat);
    if (!mine.length) continue;
    const rank = Math.round(mine.reduce((sum, r) => sum + r.rank, 0) / mine.length);
    byStat[stat] = { rank, of: mine[0].of, grade: gradeFor(rank, mine[0].of) };
  }
  const byRank = [...cells].sort((a, b) => a.rank / a.of - b.rank / b.of);
  return { byStat, softest: byRank[0], toughest: byRank[byRank.length - 1] };
}
