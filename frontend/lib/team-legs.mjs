import { fmtLine, fmtPrice } from './game-markets.mjs';

/** ESPN gives spread and total lines but no price, so these legs carry the standard price and say so. */
export const ASSUMED_PRICE = -110;

const TEAM_STATS = new Set(['ML', 'SPREAD', 'TOTAL']);
export const isTeamLeg = leg => TEAM_STATS.has(leg?.stat);
export const priceIsAssumed = leg => leg?.stat === 'SPREAD' || leg?.stat === 'TOTAL';

const base = (game, league, date) => ({ league, date, matchup: game.matchup, addedAt: new Date().toISOString() });
const abbrev = (game, side) => (side === 'home' ? game.home_team : game.away_team);

/** To win outright. The price is the real ESPN quote; null when there is none. */
export function mlLeg({ game, side, league, date, probability = null }) {
  const price = side === 'home' ? game.home_ml : game.away_ml;
  if (!Number.isFinite(price)) return null;
  const team = abbrev(game, side);
  return { ...base(game, league, date), player: team, team, stat: 'ML', direction: 'Win', line: 0, price, probability };
}

/** The handicap this team carries today (the line is the home handicap). */
export function spreadLeg({ game, side, league, date }) {
  if (!Number.isFinite(game.spread)) return null;
  const team = abbrev(game, side);
  const line = (side === 'home' ? game.spread : -game.spread) || 0; // never -0
  return { ...base(game, league, date), player: team, team, stat: 'SPREAD', direction: 'Cover', line, price: ASSUMED_PRICE, probability: null };
}

/** Over or under the game total. It belongs to the game, not a team, so the matchup stands in for the player. */
export function totalLeg({ game, direction, league, date }) {
  if (!Number.isFinite(game.total)) return null;
  return { ...base(game, league, date), player: game.matchup, stat: 'TOTAL', direction, line: game.total, price: ASSUMED_PRICE, probability: null };
}

export function legTitle(leg) {
  if (leg.stat === 'ML') return `${leg.player} moneyline`;
  if (leg.stat === 'SPREAD') return `${leg.player} ${fmtLine(leg.line)}`;
  if (leg.stat === 'TOTAL') return `${leg.direction} ${leg.line}`;
  return leg.player;
}

export function legDetail(leg) {
  if (leg.stat === 'ML') return 'To win';
  if (leg.stat === 'SPREAD') return 'Spread';
  if (leg.stat === 'TOTAL') return 'Total points';
  return `${leg.direction} ${leg.line} ${leg.stat}`;
}

/** One line for plain-text slips: what the bet is and its price, flagged when the price is assumed. */
export function legText(leg) {
  const price = `${fmtPrice(leg.price)}${priceIsAssumed(leg) ? ', assumed' : ''}`;
  if (leg.stat === 'ML') return `${legTitle(leg)} (${price})`;
  if (leg.stat === 'SPREAD') return `${legTitle(leg)} spread (${price})`;
  if (leg.stat === 'TOTAL') return `${legTitle(leg)} total (${price})`;
  return `${leg.player} ${leg.direction} ${leg.line} ${leg.stat} (${price})`;
}
