import test from 'node:test';
import assert from 'node:assert/strict';
import { ASSUMED_PRICE, isTeamLeg, priceIsAssumed, mlLeg, spreadLeg, totalLeg, legTitle, legDetail, legText } from './team-legs.mjs';
import { toggleLeg, legKey, slipText, summarizeSlip } from './bet-slip.mjs';

const game = { matchup: 'BOS @ DET', home_team: 'DET', away_team: 'BOS', spread: 1.5, total: 221.5, home_ml: 100, away_ml: -120 };
const ctx = { league: 'nba', date: '2026-10-20' };

test('a moneyline leg carries the real price and the model chance it was given', () => {
  const leg = mlLeg({ game, side: 'away', ...ctx, probability: 0.52 });
  assert.deepEqual({ player: leg.player, stat: leg.stat, direction: leg.direction, line: leg.line, price: leg.price, probability: leg.probability, matchup: leg.matchup },
    { player: 'BOS', stat: 'ML', direction: 'Win', line: 0, price: -120, probability: 0.52, matchup: 'BOS @ DET' });
  assert.equal(mlLeg({ game, side: 'home', ...ctx }).price, 100);
  assert.equal(mlLeg({ game, side: 'home', ...ctx }).probability, null);
  assert.equal(mlLeg({ game: { ...game, home_ml: null }, side: 'home', ...ctx }), null);
});

test('a spread leg takes the handicap each team carries and an assumed standard price', () => {
  assert.equal(spreadLeg({ game, side: 'home', ...ctx }).line, 1.5);
  const away = spreadLeg({ game, side: 'away', ...ctx });
  assert.deepEqual({ player: away.player, stat: away.stat, line: away.line, price: away.price, probability: away.probability }, { player: 'BOS', stat: 'SPREAD', line: -1.5, price: ASSUMED_PRICE, probability: null });
  assert.equal(spreadLeg({ game: { ...game, spread: null }, side: 'home', ...ctx }), null);
});

test('a total leg is about the game, not a team', () => {
  const over = totalLeg({ game, direction: 'Over', ...ctx });
  assert.deepEqual({ player: over.player, stat: over.stat, direction: over.direction, line: over.line, price: over.price }, { player: 'BOS @ DET', stat: 'TOTAL', direction: 'Over', line: 221.5, price: ASSUMED_PRICE });
  assert.equal(totalLeg({ game: { ...game, total: null }, direction: 'Under', ...ctx }), null);
});

test('only spread and total prices are flagged as assumed, and only team legs are team legs', () => {
  assert.equal(isTeamLeg(mlLeg({ game, side: 'home', ...ctx })), true);
  assert.equal(isTeamLeg({ stat: 'PTS' }), false);
  assert.equal(priceIsAssumed(mlLeg({ game, side: 'home', ...ctx })), false);
  assert.equal(priceIsAssumed(spreadLeg({ game, side: 'home', ...ctx })), true);
  assert.equal(priceIsAssumed(totalLeg({ game, direction: 'Over', ...ctx })), true);
  assert.equal(priceIsAssumed({ stat: 'PTS' }), false);
});

test('titles and details read like a bet slip, not like a player prop', () => {
  const ml = mlLeg({ game, side: 'away', ...ctx });
  const spread = spreadLeg({ game, side: 'away', ...ctx });
  const over = totalLeg({ game, direction: 'Over', ...ctx });
  assert.deepEqual([legTitle(ml), legDetail(ml)], ['BOS moneyline', 'To win']);
  assert.deepEqual([legTitle(spread), legDetail(spread)], ['BOS -1.5', 'Spread']);
  assert.deepEqual([legTitle(over), legDetail(over)], ['Over 221.5', 'Total points']);
  const prop = { player: 'Cade Cunningham', stat: 'PTS', direction: 'Over', line: 24.5 };
  assert.deepEqual([legTitle(prop), legDetail(prop)], ['Cade Cunningham', 'Over 24.5 PTS']);
});

test('slip text names the market and says when a price is assumed', () => {
  const legs = [mlLeg({ game, side: 'away', ...ctx }), spreadLeg({ game, side: 'home', ...ctx })];
  const text = slipText(legs, 'parlay', null);
  assert.match(text, /BOS moneyline.*-120/);
  assert.match(text, /DET \+1\.5.*assumed/);
  assert.match(legText(legs[0]), /^BOS moneyline/);
});

test('the same total toggles between over and under, while both sides of a spread can sit together', () => {
  const over = totalLeg({ game, direction: 'Over', ...ctx });
  const under = totalLeg({ game, direction: 'Under', ...ctx });
  assert.equal(legKey(over), legKey(under));
  assert.deepEqual(toggleLeg([over], under).map(l => l.direction), ['Under']);
  assert.deepEqual(toggleLeg([over], over), []);
  const home = spreadLeg({ game, side: 'home', ...ctx });
  const away = spreadLeg({ game, side: 'away', ...ctx });
  assert.notEqual(legKey(home), legKey(away));
  assert.equal(toggleLeg([home], away).length, 2);
});

test('a slip of team legs still prices a parlay from its prices', () => {
  const legs = [mlLeg({ game, side: 'away', ...ctx }), totalLeg({ game, direction: 'Over', ...ctx })];
  assert.ok(summarizeSlip(legs).american > 0);
});

test('first-quarter legs read as 1Q stats and are kept out of box-score tracking', async () => {
  const { statLabel, isQuarterLeg } = await import('./team-legs.mjs');
  const leg = { player: 'Jayson Tatum', stat: 'PTS_1Q', direction: 'Over', line: 6.5, price: -115 };
  assert.equal(legDetail(leg), 'Over 6.5 1Q PTS');
  assert.equal(legText(leg), 'Jayson Tatum Over 6.5 1Q PTS (-115)');
  assert.equal(statLabel('FG3M_1Q'), '1Q 3PM');
  assert.equal(statLabel('PTS+REB'), 'PTS+REB');
  assert.equal(isQuarterLeg(leg), true);
  assert.equal(isQuarterLeg({ ...leg, stat: 'PTS' }), false);
});
