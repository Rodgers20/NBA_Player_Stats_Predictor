/** Average CLV % over bets that carry a computed clv_pct; null when no bet has close data. */
export function averageClv(bets) {
  const values = bets.map(bet => bet?.clv?.clv_pct).filter(value => Number.isFinite(value));
  return values.length ? { count: values.length, average: values.reduce((a, b) => a + b, 0) / values.length } : null;
}
