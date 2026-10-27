import test from 'node:test';
import assert from 'node:assert/strict';
import { modifyLeg } from './slip-modify.mjs';
import { ASSUMED_PRICE } from './team-legs.mjs';

const leg = { league: 'nba', date: '2026-10-27', player: 'Jrue Holiday', stat: 'AST', direction: 'Over', line: 3.5, price: -113, probability: 0.58, matchup: 'POR @ LAL', addedAt: '' };
const prop = { player: 'Jrue Holiday', stat: 'AST', direction: 'Over', line: 3.5, model_prob: 0.58, books: [
  { book: 'FanDuel', line: 3.5, over_price: -113, under_price: -112 },
  { book: 'DraftKings', line: 3.5, over_price: -105, under_price: -125 },
  { book: 'BetMGM', line: 4.5, over_price: 150, under_price: -190 },
] };

test('switching side reprices from the best quote at that line and drops the model chance', () => {
  const under = modifyLeg(leg, prop, { direction: 'Under' });
  assert.deepEqual([under.direction, under.line, under.price, under.probability, under.assumedPrice], ['Under', 3.5, -112, null, undefined]);
  assert.equal(leg.direction, 'Over', 'the original leg is untouched');
});

test('a line a book offers takes its price; a line nobody offers assumes the standard price', () => {
  assert.deepEqual((l => [l.line, l.price, l.assumedPrice])(modifyLeg(leg, prop, { line: 4.5 })), [4.5, 150, undefined]);
  assert.deepEqual((l => [l.line, l.price, l.assumedPrice, l.probability])(modifyLeg(leg, prop, { line: 2.5 })), [2.5, ASSUMED_PRICE, true, null]);
});

test('back on the model’s side and line, its chance and the best price return', () => {
  const back = modifyLeg(modifyLeg(leg, prop, { line: 2.5 }), prop, { line: 3.5 });
  assert.deepEqual([back.price, back.probability, back.assumedPrice], [-105, 0.58, undefined]);
});

test('lines stay above zero, and a leg off the board keeps its own price', () => {
  assert.equal(modifyLeg({ ...leg, line: 0.5 }, prop, { line: -0.5 }).line, 0.5);
  const q1 = { ...leg, stat: 'PTS_1Q', price: 120 };
  assert.deepEqual((l => [l.line, l.price])(modifyLeg(q1, null, { line: 4.5 })), [4.5, 120]);
});
