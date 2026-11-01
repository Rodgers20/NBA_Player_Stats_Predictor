/** Where the model sits against the book's spread and total, so a slate card can highlight its picks the way the NFL board does. */

const Z_80 = 1.2816; // an 80% interval spans +/- 1.2816 standard deviations
const COIN_FLIP_BAND = 0.02; // a side within 2 points of 50% is a coin flip, not a pick
const FALLBACK_SIGMA = { margin: 12, total: 17 };

const finite = v => typeof v === 'number' && Number.isFinite(v);

/** Standard normal CDF via the Abramowitz-Stegun 7.1.26 erf approximation (error < 1.5e-7). */
export function normalCdf(z) {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-x * x);
  return z >= 0 ? 0.5 + erf / 2 : 0.5 - erf / 2;
}

/** Standard deviation implied by an 80% prediction interval [low, high]; a fixed fallback when it is missing. */
export function sigmaFromInterval(interval, kind) {
  if (Array.isArray(interval) && finite(interval[0]) && finite(interval[1]) && interval[1] > interval[0]) {
    return (interval[1] - interval[0]) / 2 / Z_80;
  }
  return FALLBACK_SIGMA[kind];
}

const pick = (p, high, low) => (Math.abs(p - 0.5) < COIN_FLIP_BAND ? null : p > 0.5 ? { side: high, p } : { side: low, p: 1 - p });

/**
 * The model's spread and total picks for one game.
 * `game.spread` is the book's home handicap (home favoured is negative); `prediction.spread` is the model's
 * projected home margin (positive = home wins by that much), so home covers when margin + handicap > 0.
 * Returns `{ spread: {side:'home'|'away', p} | null, total: {side:'over'|'under', p} | null }`.
 */
export function modelLeans(game, prediction) {
  const intervals = prediction?.prediction_intervals;
  const spread = finite(game?.spread) && finite(prediction?.spread)
    ? pick(normalCdf((prediction.spread + game.spread) / sigmaFromInterval(intervals?.margin, 'margin')), 'home', 'away') : null;
  const total = finite(game?.total) && finite(prediction?.total)
    ? pick(normalCdf((prediction.total - game.total) / sigmaFromInterval(intervals?.total, 'total')), 'over', 'under') : null;
  return { spread, total };
}
