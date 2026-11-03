/** Pure helpers for the NHL pages. Probabilities are 0-1; odds are American integers. */

export const MARKETS = [
  { key: 'goal', label: 'Anytime goal', short: 'Goal', stat: 'goals', unit: 'goals' },
  { key: 'sog', label: 'Shots on goal', short: 'SOG', stat: 'sog', unit: 'shots' },
  { key: 'assists', label: 'Assists', short: 'Ast', stat: 'assists', unit: 'assists' },
  { key: 'points', label: 'Points', short: 'Pts', stat: 'points', unit: 'points' },
];

export const CONFIDENCE_LABEL = { high: 'High confidence', medium: 'Medium confidence', low: 'Low confidence' };

/** "+126" / "-127"; null when there is no price. */
export function american(odds) {
  if (odds == null || !Number.isFinite(odds)) return null;
  return odds > 0 ? `+${Math.round(odds)}` : `${Math.round(odds)}`;
}

/** The fair (no-vig) American price for a probability: 0.6 -> -150, 0.4 -> +150. */
export function fairOdds(p) {
  if (p == null || !(p > 0 && p < 1)) return null;
  return p >= 0.5 ? Math.round((-100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p);
}

export const pct = (value, digits = 0) => (value == null ? '—' : `${(value * 100).toFixed(digits)}%`);
export const num = (value, digits = 1) => (value == null ? '—' : value.toFixed(digits));
export const signed = (value, digits = 1) => (value == null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(digits)}`);

/** "7:00 PM ET" for a UTC ISO start. */
export function tipText(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return 'TBA';
  return `${d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/New_York' })} ET`;
}

/** Group key for the slate: the Eastern hour of the puck drop. */
export function hourLabel(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return 'TBA';
  return `${d.toLocaleTimeString('en-US', { hour: 'numeric', timeZone: 'America/New_York' })} ET`;
}

/** "Final", "Final / OT", "Final / SO", "Live" or the puck-drop time. */
export function statusText(game) {
  if (game.status === 'final') return game.period_type === 'OT' || game.period_type === 'SO' ? `Final / ${game.period_type}` : 'Final';
  if (game.status === 'live') return 'Live';
  return tipText(game.start);
}

/** Home-team win probability bar colours; the contract carries no team colours, so sides get two fixed hues. */
export const SIDE_COLOR = { away: '#7aa7b0', home: '#f5ba64' };

export const marketOf = key => MARKETS.find(m => m.key === key);

/** "Over 2.5" / "1+" style label for a line. Anytime goal reads "Anytime"; whole-number thresholds read "N+". */
export function lineLabel(market, line) {
  if (market === 'goal') return 'Anytime';
  return `${Math.floor(line) + 1}+`;
}

/** Stat value of a game-log row for a market. */
export function logValue(row, market) {
  const stat = marketOf(market)?.stat;
  return stat ? row[stat] ?? null : null;
}

/** Games for a chart window, newest first in, oldest first out. `n` = 0 means the whole current season (first `seasonGp` rows). */
export function windowGames(log, n, seasonGp) {
  const take = n === 0 ? Math.min(seasonGp ?? log.length, log.length) : n;
  return log.slice(0, take).reverse();
}

/** Hits of a stat at/over a threshold: `line` is x.5, so over means value > line. */
export function hitSummary(values, line) {
  const real = values.filter(v => v != null);
  const hits = real.filter(v => v > line).length;
  return { n: real.length, hits, rate: real.length ? hits / real.length : null };
}

/** Lines to offer for a market on the analysis page, taken from the player's own hit-rate table. */
export function linesFor(detail, market) {
  return (detail.hit_rates?.[market] ?? []).map(r => r.line);
}

/** Pick the line closest to a coin flip as the default for a market with several. */
export function defaultLine(lines) {
  if (!lines.length) return null;
  return lines.includes(2.5) ? 2.5 : lines.includes(0.5) ? 0.5 : lines[0];
}
