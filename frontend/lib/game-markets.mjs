import { marketForm } from './market-form.mjs';

/** American odds with an explicit sign; missing is a dash, never zero. */
export const fmtPrice = price => (price == null ? '—' : price > 0 ? `+${price}` : `${price}`);
/** Spread-style line: +1.5, -3, PK. */
export const fmtLine = line => (line === 0 ? 'PK' : line > 0 ? `+${line}` : `${line}`);

/**
 * The three main markets for a game as display-ready sides, each with recent form against today's number.
 * A market with no line is null so the UI can say "No line yet". Spread is the home handicap.
 */
export function gameMarkets(game, awayAbbrev, homeAbbrev, awayGames, homeGames) {
  const form = marketForm(game, homeGames, awayGames);
  const moneyline = game.away_ml == null && game.home_ml == null ? null : [
    { label: awayAbbrev, value: fmtPrice(game.away_ml), form: { ...form.moneyline.away, caption: 'L10 wins', flat: true, note: 'straight up' } },
    { label: homeAbbrev, value: fmtPrice(game.home_ml), form: { ...form.moneyline.home, caption: 'L10 wins', flat: true, note: 'straight up' } },
  ];
  const spread = game.spread == null || !form.spread ? null : [
    { label: awayAbbrev, value: fmtLine(-game.spread), form: { ...form.spread.away, caption: 'L10 covers', flat: true, note: `vs ${fmtLine(-game.spread)}` } },
    { label: homeAbbrev, value: fmtLine(game.spread), form: { ...form.spread.home, caption: 'L10 covers', flat: true, note: `vs ${fmtLine(game.spread)}` } },
  ];
  const total = game.total == null || !form.total ? null : [
    { label: awayAbbrev, form: { values: form.total.away, line: game.total, caption: 'L10 over', note: `vs ${game.total}` } },
    { label: homeAbbrev, form: { values: form.total.home, line: game.total, caption: 'L10 over', note: `vs ${game.total}` } },
  ];
  return { moneyline, spread, total, totalTitle: game.total == null ? 'Total' : `Total ${game.total}` };
}
