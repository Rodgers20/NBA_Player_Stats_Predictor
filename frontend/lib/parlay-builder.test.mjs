import test from 'node:test';
import assert from 'node:assert/strict';
import { buildParlay, canBuildFrom, propLeg } from './parlay-builder.mjs';

const prop = (player, game, over) => ({
  player, stat: 'PTS', direction: 'Over', line: 20.5, price: -110, model_prob: 0.6, edge_pct: 5, ev: 0.1,
  pick_type: 'pick', has_live_odds: true, game_matchup: game, team: 'AAA', ...over,
});

const board = [
  prop('A', 'G1', { model_prob: 0.80, edge_pct: 2, ev: 0.05 }),
  prop('A', 'G1', { stat: 'REB', model_prob: 0.75 }),
  prop('B', 'G1', { model_prob: 0.70, edge_pct: 9, ev: 0.30 }),
  prop('C', 'G2', { model_prob: 0.65, edge_pct: 4, ev: 0.12 }),
  prop('D', 'G3', { model_prob: 0.60, edge_pct: 7, ev: 0.20 }),
  prop('R', 'G4', { model_prob: 0.99, pick_type: 'research' }),
  prop('N', 'G5', { model_prob: 0.95, price: null }),
];

test('only priced model picks are eligible', () => {
  assert.equal(canBuildFrom(board[0]), true);
  assert.equal(canBuildFrom(board[5]), false, 'research rows are not picks');
  assert.equal(canBuildFrom(board[6]), false, 'no price, no leg');
  assert.equal(canBuildFrom(prop('X', 'G', { direction: 'Yes' })), false);
  assert.equal(canBuildFrom(prop('X', 'G', { model_prob: 1 })), false);
});

test('safe takes the most likely legs, one per player, spreading across games first', () => {
  const legs = buildParlay(board, { legs: 3, risk: 'safe' });
  assert.deepEqual(legs.map(p => [p.player, p.stat]), [['A', 'PTS'], ['C', 'PTS'], ['D', 'PTS']]);
});

test('same game is used only once distinct games run out', () => {
  const legs = buildParlay(board, { legs: 4, risk: 'safe' });
  assert.deepEqual(legs.map(p => p.player), ['A', 'C', 'D', 'B']);
});

test('edge and payout rank by their own measure', () => {
  assert.deepEqual(buildParlay(board, { legs: 2, risk: 'edge' }).map(p => p.player), ['B', 'D']);
  assert.deepEqual(buildParlay(board, { legs: 2, risk: 'payout' }).map(p => p.player), ['B', 'D']);
});

test('required players come first and never exceed the leg count', () => {
  assert.deepEqual(buildParlay(board, { legs: 2, risk: 'safe', include: ['D'] }).map(p => p.player), ['D', 'A']);
  assert.deepEqual(buildParlay(board, { legs: 1, risk: 'safe', include: ['D', 'C'] }).map(p => p.player), ['D']);
  assert.deepEqual(buildParlay(board, { legs: 2, risk: 'safe', include: ['Nobody'] }).map(p => p.player), ['A', 'C']);
});

test('a thin board returns what it has, not a padded slip', () => {
  assert.equal(buildParlay(board, { legs: 6, risk: 'safe' }).length, 4);
  assert.deepEqual(buildParlay([], { legs: 3 }), []);
});

test('the input board is not reordered', () => {
  const copy = [...board];
  buildParlay(board, { legs: 3, risk: 'edge' });
  assert.deepEqual(board, copy);
});

test('a prop becomes the same slip leg the board row adds', () => {
  const leg = propLeg(board[2], 'nba', '2026-10-24');
  assert.deepEqual({ ...leg, addedAt: '' }, {
    league: 'nba', date: '2026-10-24', player: 'B', stat: 'PTS', direction: 'Over', line: 20.5, price: -110,
    probability: 0.70, matchup: 'G1', team: 'AAA', addedAt: '',
  });
});
