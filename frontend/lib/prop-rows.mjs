import { hitChips, hitTone } from './prop-research.mjs';

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
  if (!(prop.total > 0) || prop.hit_rate == null || !Number.isFinite(prop.hit_rate)) return [];
  const pct = Math.round(prop.hit_rate);
  return [{ key: 'all', label: 'All', hits: prop.hits, total: prop.total, pct, tone: hitTone(pct) }];
}
