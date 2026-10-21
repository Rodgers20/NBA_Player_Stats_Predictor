import { marketForm } from './market-form.mjs';

/** American odds with an explicit sign; missing is a dash, never zero. */
export const fmtPrice = price => (price == null ? '—' : price > 0 ? `+${price}` : `${price}`);
/** Spread-style line: +1.5, -3, PK. */
export const fmtLine = line => (line === 0 ? 'PK' : line > 0 ? `+${line}` : `${line}`);

/** Prefix each caption with the number of games actually used ("L7 wins"), so short samples never claim L10. */
const windowed = pair => pair && pair.map(side => (side.form ? { ...side, form: { ...side.form, caption: `L${side.form.values.length} ${side.form.caption}` } } : side));

/**
 * The three main markets for a game as display-ready sides, each with recent form against today's number.
 * A market with no line is null so the UI can say "No line yet". Spread is the home handicap.
 */
export function gameMarkets(game, awayAbbrev, homeAbbrev, awayGames, homeGames) {
  const form = marketForm(game, homeGames, awayGames);
  const moneyline = game.away_ml == null && game.home_ml == null ? null : [
    { label: awayAbbrev, value: fmtPrice(game.away_ml), form: { ...form.moneyline.away, caption: 'wins', flat: true, note: 'straight up' } },
    { label: homeAbbrev, value: fmtPrice(game.home_ml), form: { ...form.moneyline.home, caption: 'wins', flat: true, note: 'straight up' } },
  ];
  const spread = game.spread == null || !form.spread ? null : [
    { label: awayAbbrev, value: fmtLine(-game.spread), form: { ...form.spread.away, caption: 'would cover', flat: true, note: `vs ${fmtLine(-game.spread)}` } },
    { label: homeAbbrev, value: fmtLine(game.spread), form: { ...form.spread.home, caption: 'would cover', flat: true, note: `vs ${fmtLine(game.spread)}` } },
  ];
  const total = game.total == null || !form.total ? null : [
    { label: awayAbbrev, form: { values: form.total.away, line: game.total, caption: 'over', note: `vs ${game.total}` } },
    { label: homeAbbrev, form: { values: form.total.home, line: game.total, caption: 'over', note: `vs ${game.total}` } },
  ];
  return { moneyline: windowed(moneyline), spread: windowed(spread), total: windowed(total), totalTitle: game.total == null ? 'Total' : `Total ${game.total}` };
}
