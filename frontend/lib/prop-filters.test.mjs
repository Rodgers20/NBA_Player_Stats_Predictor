import test from 'node:test';
import assert from 'node:assert/strict';
import { forBooks, applyBoardFilters, availableBooks, availablePlayers, toggleStat, selectedStats } from './prop-filters.mjs';

test('stat chips toggle in and out of a multi-selection', () => {
  assert.equal(toggleStat('', 'PTS'), 'PTS');
  assert.equal(toggleStat('PTS', 'AST'), 'PTS,AST');
  assert.equal(toggleStat('PTS,AST', 'PTS'), 'AST');
  assert.equal(toggleStat('AST', 'AST'), '', 'removing the last stat means All');
  assert.equal(toggleStat('PTS,AST', ''), '', 'All clears the selection');
});

test('a whole-game line market replaces the player stats and toggles off on its own', () => {
  assert.equal(toggleStat('PTS,AST', 'ML'), 'ML');
  assert.equal(toggleStat('ML', 'PTS'), 'PTS');
  assert.equal(toggleStat('ML', 'ML'), '');
});

test('the selection reads back as a list', () => {
  assert.deepEqual(selectedStats(''), []);
  assert.deepEqual(selectedStats('PTS,PTS+AST'), ['PTS', 'PTS+AST']);
  assert.deepEqual(selectedStats(undefined), []);
});

const prop = (over = {}) => ({
  player: 'A', stat: 'PTS', direction: 'Over', line: 10.5, price: -110, ev: 0.1, model_prob: 0.6, pick_type: 'pick',
  stars: 4, edge_pct: 5,
  books: [
    { book: 'FanDuel', line: 10.5, over_price: -110, under_price: -110 },
    { book: 'DraftKings', line: 10.5, over_price: 100, under_price: -120 },
    { book: 'BetMGM', line: 11.5, over_price: 120, under_price: -140 },
  ],
  ...over,
});

test('no chosen books leaves the pick untouched', () => {
  const original = prop();
  assert.equal(forBooks(original, []), original);
  assert.equal(forBooks(original, undefined), original);
});

test('a pick needs a chosen book quoting the same line and a price on its side', () => {
  assert.equal(forBooks(prop(), ['BetMGM']), null, 'BetMGM posts a different line');
  assert.equal(forBooks(prop(), ['Caesars']), null, 'no quote at all');
  assert.equal(forBooks(prop({ direction: 'Under', books: [{ book: 'FanDuel', line: 10.5, over_price: -110, under_price: null }] }), ['FanDuel']), null);
});

test('the best chosen price is shown and EV is recomputed from the model probability', () => {
  const adjusted = forBooks(prop(), ['FanDuel', 'DraftKings']);
  assert.equal(adjusted.price, 100); assert.equal(adjusted.book, 'DraftKings');
  assert.ok(Math.abs(adjusted.ev - 0.2) < 1e-9, 'p 0.6 at even money: 0.6 * 1 - 0.4');
  assert.equal(adjusted.pick_type, 'pick');
  assert.equal(adjusted.stars, null); assert.equal(adjusted.edge_pct, null, 'stale edge is not kept for a new price');
});

test('under picks use the under price', () => {
  const adjusted = forBooks(prop({ direction: 'Under', price: -120, model_prob: 0.6 }), ['FanDuel']);
  assert.equal(adjusted.price, -110);
  assert.ok(Math.abs(adjusted.ev - (0.6 * (100 / 110) - 0.4)) < 1e-9);
});

test('a price that leaves no edge turns a pick into a lean', () => {
  const adjusted = forBooks(prop({ model_prob: 0.5, books: [{ book: 'FanDuel', line: 10.5, over_price: -150, under_price: 130 }] }), ['FanDuel']);
  assert.ok(adjusted.ev < 0); assert.equal(adjusted.pick_type, 'lean');
});

test('whole-number lines can push, so a new price drops the EV instead of guessing one', () => {
  const adjusted = forBooks(prop({ line: 10, books: [{ book: 'DraftKings', line: 10, over_price: 100, under_price: -120 }] }), ['DraftKings']);
  assert.equal(adjusted.price, 100); assert.equal(adjusted.ev, null); assert.equal(adjusted.pick_type, 'lean');
});

test('the same price keeps the original EV and edge', () => {
  const adjusted = forBooks(prop(), ['FanDuel']);
  assert.equal(adjusted.ev, 0.1); assert.equal(adjusted.stars, 4); assert.equal(adjusted.book, 'FanDuel');
});

const board = [
  prop({ player: 'A', stat: 'PTS' }),
  prop({ player: 'B', stat: 'AST', pick_type: 'lean', ev: -0.02, model_prob: 0.45 }),
  prop({ player: 'C', stat: 'PTS+REB' }),
];

test('no filters returns every pick in order', () => {
  assert.deepEqual(applyBoardFilters(board, {}).map(p => p.player), ['A', 'B', 'C']);
});

test('only +EV keeps real picks', () => {
  assert.deepEqual(applyBoardFilters(board, { evOnly: true }).map(p => p.player), ['A', 'C']);
});

test('stats and players narrow the board together', () => {
  assert.deepEqual(applyBoardFilters(board, { stats: ['PTS', 'AST'] }).map(p => p.player), ['A', 'B']);
  assert.deepEqual(applyBoardFilters(board, { stats: ['PTS', 'AST'], players: ['B', 'C'] }).map(p => p.player), ['B']);
});

test('books filter and re-price in the same pass', () => {
  const result = applyBoardFilters(board, { books: ['DraftKings'], evOnly: true });
  assert.deepEqual(result.map(p => p.player), ['A', 'C']);
  assert.ok(result.every(p => p.book === 'DraftKings' && p.price === 100));
});

test('available books and players are unique and sorted', () => {
  assert.deepEqual(availableBooks(board), ['BetMGM', 'DraftKings', 'FanDuel']);
  assert.deepEqual(availablePlayers([...board, prop({ player: 'A' })]), ['A', 'B', 'C']);
});
