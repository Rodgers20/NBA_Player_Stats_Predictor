import { decimalOdds } from './bet-slip.mjs';
import { marketProbability } from './game-board.mjs';

/** A moneyline edge below this is inside the noise of a model estimate, so it is not called bettable. */
export const EDGE_MIN = 0.02;

/** Expected profit per unit staked: p*(d-1) - (1-p). Null without a valid price or a probability strictly between 0 and 1. */
export function evPerUnit(prob, price) {
  const decimal = decimalOdds(price);
  return decimal !== null && prob > 0 && prob < 1 ? prob * (decimal - 1) - (1 - prob) : null;
}

/**
 * Moneyline value for every pregame side that has both a real price and a model win chance, best edge first.
 * `marketProb` is the sportsbook's chance with the margin removed, so the model and the market can be compared.
 */
export function gameEdges(games, predictions) {
  const byMatchup = new Map(predictions.map(item => [item.matchup, item]));
  const rows = [];
  for (const game of games) {
    if (game.status_state && game.status_state !== 'pre') continue;
    const home = byMatchup.get(game.matchup)?.home_win_probability;
    if (!(home > 0 && home < 1)) continue;
    const market = marketProbability(game);
    for (const side of ['home', 'away']) {
      const price = side === 'home' ? game.home_ml : game.away_ml;
      const modelProb = side === 'home' ? home : 1 - home;
      const ev = evPerUnit(modelProb, price);
      if (ev === null) continue;
      rows.push({
        game, side, team: side === 'home' ? game.home_team : game.away_team, price, modelProb, ev,
        marketProb: market == null ? null : side === 'home' ? market : 1 - market,
      });
    }
  }
  return rows.sort((a, b) => b.ev - a.ev);
}

export const bettableEdges = rows => rows.filter(row => row.ev >= EDGE_MIN);
