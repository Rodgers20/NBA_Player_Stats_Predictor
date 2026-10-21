import test from 'node:test';
import assert from 'node:assert/strict';
import { windowGames, gradeGames, hitSummary, average } from './chart-window.mjs';

const games = [
  { opponent: 'A vs. X', season: '2025', value: 4 }, { opponent: 'A @ Y', season: '2026', value: 6 },
  { opponent: 'A vs. X', season: '2026', value: 5 }, { opponent: 'A @ Z', season: '2026', value: null },
];

test('windows slice recent games, head-to-head and the latest season', () => {
  assert.equal(windowGames(games, 'L5').length, 4);
  assert.deepEqual(windowGames(games, 'L5').slice(-2), games.slice(-2));
  assert.deepEqual(windowGames(games, 'H2H', 'X').map(g => g.value), [4, 5]);
  assert.deepEqual(windowGames(games, 'H2H', ''), []);
  assert.deepEqual(windowGames(games, 'Season').map(g => g.value), [6, 5, null]);
});

test('grading marks hits, misses, pushes and missing values without mutating input', () => {
  const graded = gradeGames(games, 5, 'over');
  assert.deepEqual(graded.map(g => g.hit), [false, true, null, null]);
  assert.deepEqual(gradeGames(games, 5, 'Under').map(g => g.hit), [true, false, null, null]);
  assert.ok(!('hit' in games[0]));
  assert.deepEqual(gradeGames(games, null).map(g => g.hit), [null, null, null, null]);
  assert.deepEqual(hitSummary(graded), { hits: 1, total: 2 });
  assert.equal(average(games), 5);
});
