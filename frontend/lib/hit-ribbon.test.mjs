import test from 'node:test';
import assert from 'node:assert/strict';
import { ribbonCells, ribbonSummary } from './hit-ribbon.mjs';

test('summary counts hits out of decided games, leaving pushes out of the denominator', () => {
  assert.deepEqual(ribbonSummary([2, 1, 0], 1), { hits: 1, decided: 2, pushes: 1 });
  assert.deepEqual(ribbonSummary([], 1), { hits: 0, decided: 0, pushes: 0 });
});

test('ribbon cells grade each game and keep oldest-first order', () => {
  assert.deepEqual(ribbonCells([2, 1, 0], 0.5).map(cell => cell.result), ['hit', 'hit', 'miss']);
  assert.equal(ribbonCells([1], 1)[0].result, 'push');
});

test('ribbon bar heights scale to the tallest of values and line, never below 4', () => {
  const cells = ribbonCells([10, 5, 0], 2);
  assert.deepEqual(cells.map(cell => cell.heightPct), [100, 50, 4]);
  assert.equal(ribbonCells([1], 4)[0].heightPct, 25);
});

test('ribbon honours the Under side', () => {
  assert.deepEqual(ribbonCells([0, 3], 1.5, 'Under').map(cell => cell.result), ['hit', 'miss']);
});

test('a mixed hit, miss and push sample reports the push-free rate inputs', () => {
  assert.deepEqual(ribbonSummary([2, 1, 0, 3], 1), { hits: 2, decided: 3, pushes: 1 });
});
