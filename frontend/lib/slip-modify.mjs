import { ASSUMED_PRICE } from './team-legs.mjs';

/**
 * A copy of the leg on a new side and/or line, repriced from the best quote at that line (the standard price when no
 * book offers it). The model chance stays only on the model's own side and line. A leg off the board keeps its price.
 */
export function modifyLeg(leg, prop, { direction = leg.direction, line = leg.line } = {}) {
  const next = { ...leg, direction, line: line > 0 ? line : leg.line };
  if (!prop) return next;
  const priceOf = b => (next.direction === 'Under' ? b.under_price : b.over_price);
  const best = (prop.books ?? []).filter(b => b.line === next.line && Number.isFinite(priceOf(b)))
    .reduce((a, b) => (a && priceOf(a) >= priceOf(b) ? a : b), null);
  const modelSide = prop.direction === next.direction && prop.line === next.line && Number.isFinite(prop.model_prob);
  const priced = { ...next, price: best ? priceOf(best) : ASSUMED_PRICE, assumedPrice: !best, probability: modelSide ? prop.model_prob : null };
  if (best) delete priced.assumedPrice;
  return priced;
}
