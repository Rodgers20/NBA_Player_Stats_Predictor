import { hitChips, hitTone } from './prop-research.mjs';
import { gradeSide } from './hit-rate.mjs';

/** The last 10 results as graded per-game cells, oldest first. [] without a line or results, never a row of misses. */
export function rowDashes(prop) {
  const values = (prop.l10_values ?? []).filter(value => value != null).slice(0, 10); // newest first
  if (prop.line == null || !values.length) return [];
  const side = /^under$/i.test(prop.direction) ? 'Under' : 'Over';
  return values.reverse().map(value => ({ value, result: gradeSide(value, prop.line, side) }));
}

/** One hit window (L5, L10, L20, h2h, season) as a chip; null when it has no games. */
export function windowCell(prop, key) {
  return hitChips(prop.hit_windows).find(chip => chip.key === key) ?? null;
}

/** Highest hit rate in the chosen window first, larger samples breaking ties, unknown windows last. Input is not mutated. */
export function sortByWindow(props, key) {
  const score = prop => windowCell(prop, key);
  return [...props].sort((a, b) => {
    const left = score(a), right = score(b);
    if (!left || !right) return Number(Boolean(right)) - Number(Boolean(left));
    return right.pct - left.pct || right.total - left.total;
  });
}

/** Plain, case-insensitive substring match on player, team and opponent. Never builds a regex. */
export function searchProps(props, query) {
  const needle = (query ?? '').trim().toLowerCase();
  if (!needle) return props;
  return props.filter(prop => [prop.player, prop.team, prop.opponent].some(field => (field ?? '').toLowerCase().includes(needle)));
}

const ROW_WINDOWS = new Set(['L5', 'L10', 'L20']);

/**
 * Hit cells for a board row: L5, L10 and L20 as hits out of games. Without windows it falls back to the
 * overall sample, and to nothing when there is none, so an unknown rate is never drawn as 0%.
 */
export function rowCells(prop) {
  const cells = hitChips(prop.hit_windows).filter(chip => ROW_WINDOWS.has(chip.label));
  if (cells.length) return cells;
  if (!(prop.total > 0) || !Number.isFinite(prop.hits) || prop.hit_rate == null || !Number.isFinite(prop.hit_rate)) return [];
  const pct = Math.round(prop.hit_rate);
  return [{ key: 'all', label: 'All', hits: prop.hits, total: prop.total, pct, tone: hitTone(pct) }];
}
