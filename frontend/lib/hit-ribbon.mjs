import { gradeSide, hitSummary } from './hit-rate.mjs';

/** One bar per game, oldest first. Height is relative to the tallest of the results and the line. */
export function ribbonCells(values, line, side = 'Over') {
  const top = Math.max(line, ...values) || 1;
  return values.map(value => ({
    value,
    result: gradeSide(value, line, side),
    heightPct: Math.max(4, Math.round((value / top) * 100)),
  }));
}

/** Hits out of decided games; pushes are refunded so they are left out of the denominator. */
export function ribbonSummary(values, line, side = 'Over') {
  const { hits, misses, pushes } = hitSummary(values, line, side);
  return { hits, decided: hits + misses, pushes };
}

/** Hit counts over the most recent N games; `n` is how many games exist, so short samples stay honest. */
export function windowSummaries(values, line, side = 'Over', windows = [5, 10, 20]) {
  return windows.map(size => {
    const { n, hits, misses, rate } = hitSummary(values.slice(-size), line, side);
    return { label: `L${size}`, n, hits, decided: hits + misses, rate };
  });
}
