import test from 'node:test';
import assert from 'node:assert/strict';
import { gradeSide, hitSummary, gradeBet } from './hit-rate.mjs';

test('a side hits, misses, or pushes against the line', () => {
  assert.equal(gradeSide(25, 24.5, 'Over'), 'hit');
  assert.equal(gradeSide(24, 24.5, 'Over'), 'miss');
  assert.equal(gradeSide(24, 24, 'Over'), 'push');
  assert.equal(gradeSide(24, 24.5, 'Under'), 'hit');
  assert.equal(gradeSide(25, 24.5, 'Under'), 'miss');
  assert.equal(gradeSide(24, 24, 'Under'), 'push');
});

test('summary excludes pushes from the rate and reports the most recent results', () => {
  const summary = hitSummary([30, 20, 25, 25, 10, 26], 25, 'Over', 3);
  assert.deepEqual({ n: summary.n, hits: summary.hits, misses: summary.misses, pushes: summary.pushes },
    { n: 6, hits: 2, misses: 2, pushes: 2 });
  assert.equal(summary.rate, 50);
  assert.deepEqual(summary.recent.map(item => item.result), ['push', 'miss', 'hit']);
  assert.deepEqual(hitSummary([], 20, 'Over').rate, null);
  assert.equal(hitSummary([25, 25], 25, 'Over').rate, null);
});

test('a bet is graded from the game played on its date, and stays open without one', () => {
  const games = [{ date: '2026-10-16', value: 12 }, { date: '2026-10-17', value: 24 }, { date: '2026-10-18', value: null }];
  assert.equal(gradeBet({ game_date: '2026-10-17', side: 'Over', line: 23.5 }, games), 'win');
  assert.equal(gradeBet({ game_date: '2026-10-17', side: 'Over', line: 24.5 }, games), 'loss');
  assert.equal(gradeBet({ game_date: '2026-10-17', side: 'Under', line: 24 }, games), 'push');
  assert.equal(gradeBet({ game_date: '2026-10-18', side: 'Over', line: 1.5 }, games), null);
  assert.equal(gradeBet({ game_date: '2026-10-19', side: 'Over', line: 1.5 }, games), null);
});
