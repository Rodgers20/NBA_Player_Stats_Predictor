// Pure stat logic for the player analysis page: windows, hit rates, lines, fair odds.
// Game rows follow soccer/docs/props-contract.md (`players/gamelog/<key>`), newest first.

export const STATS = {
  goals:   { label: "Goals", lines: [0.5, 1.5], defaultLine: 0.5, market: "anytime_goal" },
  assists: { label: "Assists", lines: [0.5, 1.5], defaultLine: 0.5, market: "assist" },
  shots:   { label: "Shots", lines: [0.5, 1.5, 2.5, 3.5, 4.5], defaultLine: 1.5, market: null },
  sot:     { label: "Shots on target", lines: [0.5, 1.5, 2.5], defaultLine: 0.5, market: "shots_on_target" },
  fouls:   { label: "Fouls", lines: [0.5, 1.5, 2.5], defaultLine: 1.5, market: null },
  cards:   { label: "Cards", lines: [0.5], defaultLine: 0.5, market: null },
};
export const STAT_IDS = Object.keys(STATS);
export const WINDOWS = ["L5", "L10", "L20", "Season", "Home", "Away", "VS"];

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** One game's value for a stat; null when the box score does not carry it. */
export function statValue(game, id) {
  switch (id) {
    case "goals": return num(game.goals);
    case "assists": return num(game.assists);
    case "shots": return num(game.shots);
    case "sot": return num(game.sot);
    case "fouls": return num(game.fouls_committed);
    case "cards": {
      const y = num(game.yellow);
      const r = num(game.red);
      return y === null && r === null ? null : (y ?? 0) + (r ?? 0);
    }
    default: return null;
  }
}

/** Games inside a window (input is newest first, output keeps that order). */
export function filterGames(games, win, opponent = null) {
  switch (win) {
    case "L5": return games.slice(0, 5);
    case "L10": return games.slice(0, 10);
    case "L20": return games.slice(0, 20);
    case "Home": return games.filter((g) => g.is_home === true);
    case "Away": return games.filter((g) => g.is_home === false);
    case "VS":
      return opponent ? games.filter((g) => g.opponent?.toLowerCase() === opponent.toLowerCase()) : [];
    default: return games.slice();
  }
}

/** Values for a stat across games, dropping games where it is missing. */
export const valuesOf = (games, id) =>
  games.map((g) => statValue(g, id)).filter((v) => v !== null);

/** Hit rate against a line. A value equal to the line is a push and counts as no hit. */
export function hitRate(values, line, side = "over") {
  const total = values.length;
  let hits = 0;
  let pushes = 0;
  for (const v of values) {
    if (v === line) pushes += 1;
    else if (side === "over" ? v > line : v < line) hits += 1;
  }
  return { hits, pushes, total, rate: total ? hits / total : null };
}

export const average = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null);

export function median(values) {
  if (!values.length) return null;
  const s = values.slice().sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function fmtNum(v, dp = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number.isInteger(v) ? String(v) : v.toFixed(dp);
}

/** "62% — 5 of 8 games over 0.5" */
export function hitHeadline(hr, line, side = "over") {
  if (!hr.total) return "No games in this window";
  return `${Math.round(hr.rate * 100)}% — ${hr.hits} of ${hr.total} games ${side} ${line}`;
}

/** Fair price for a probability: decimal and American odds, null outside (0,1). */
export function fairOdds(prob) {
  if (typeof prob !== "number" || !(prob > 0 && prob < 1)) return null;
  const decimal = Math.round((1 / prob) * 100) / 100;
  const american = prob >= 0.5
    ? `-${Math.round((100 * prob) / (1 - prob))}`
    : `+${Math.round((100 * (1 - prob)) / prob)}`;
  return { decimal, american };
}

/** The model's probability for "stat side line", from this player's next-game prop rows. */
export function modelProbFor(props, id, line, side = "over") {
  const market = STATS[id]?.market;
  if (!market || !Array.isArray(props)) return null;
  const row = props.find((p) => p.market === market && (market === "shots_on_target" ? p.line === line : line === 0.5));
  if (!row) return null;
  return { prob: side === "over" ? row.prob : Math.round((1 - row.prob) * 1e4) / 1e4, prop: row };
}

/** Closest allowed line to a requested one (for URL / state recovery). */
export function nearestLine(id, line) {
  const lines = STATS[id].lines;
  return lines.reduce((best, l) => (Math.abs(l - line) < Math.abs(best - line) ? l : best), lines[0]);
}
