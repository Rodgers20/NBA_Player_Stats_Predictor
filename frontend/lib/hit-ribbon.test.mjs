import test from 'node:test';
import assert from 'node:assert/strict';
import { ribbonCells, windowSummaries } from './hit-ribbon.mjs';

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

test('windows count the most recent games and report hits out of n', () => {
  const windows = windowSummaries([1, 0, 1, 1, 0, 1], 0.5);
  const l5 = windows.find(item => item.label === 'L5');
  assert.deepEqual({ n: l5.n, hits: l5.hits, rate: l5.rate }, { n: 5, hits: 3, rate: 60 });
  const l10 = windows.find(item => item.label === 'L10');
  assert.equal(l10.n, 6);
});

test('windows with no games or only pushes have a null rate, never zero', () => {
  assert.equal(windowSummaries([], 0.5)[0].rate, null);
  assert.equal(windowSummaries([], 0.5)[0].n, 0);
  assert.equal(windowSummaries([1, 1], 1)[0].rate, null);
});
