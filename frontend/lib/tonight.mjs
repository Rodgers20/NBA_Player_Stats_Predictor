/** Win rate a bet at this American price must hit to break even, in percent. */
export function breakEven(price) {
  if (!Number.isFinite(price) || price === 0) return null;
  return price < 0 ? (-price / (-price + 100)) * 100 : (100 / (price + 100)) * 100;
}

const STALE_MINUTES = 60;

/** How old a sportsbook quote is; null when the timestamp is missing or unreadable. */
export function quoteAge(updatedAt, now = Date.now()) {
  const stamp = updatedAt ? Date.parse(updatedAt) : NaN;
  if (Number.isNaN(stamp)) return null;
  const minutes = Math.max(0, Math.round((now - stamp) / 60000));
  const label = minutes < 1 ? 'just now' : minutes < 60 ? `${minutes} min ago` : `${Math.round(minutes / 60)} hr ago`;
  return { label, stale: minutes > STALE_MINUTES };
}

const EARLY_RECORD = 30;
const comboKey = stat => stat.split('+').sort().join('+');

/** The graded record for a stat, matching combos in any part order; null when nothing is graded. `early` flags a small sample. */
export function recordFor(record, stat) {
  const match = Object.entries(record?.by_stat ?? {}).find(([key]) => comboKey(key) === comboKey(stat));
  if (!match || !match[1].total) return null;
  const { hit, total, pct } = match[1];
  return { hit, total, pct, early: total < EARLY_RECORD };
}

/** Splits a props response into what can be bet (picks), priced-but-no-edge (leans), and unpriced model research. */
export function tonightBoard(props) {
  const byEv = (a, b) => (b.ev ?? -Infinity) - (a.ev ?? -Infinity);
  return {
    picks: props.filter(prop => prop.pick_type === 'pick').sort(byEv),
    leans: props.filter(prop => prop.pick_type === 'lean').sort(byEv),
    research: props.filter(prop => prop.pick_type === 'research'),
  };
}
