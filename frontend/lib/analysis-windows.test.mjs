import test from 'node:test';
import assert from 'node:assert/strict';
import { selectWindow, windowCounts } from './analysis-windows.mjs';

test('window counts grade an Under prop as hits below the line', () => {
  const games = [10, 20, 30].map((value, i) => ({ date: `2026-05-0${i + 1}`, opponent: 'BOS', value, season: '2025-26' }));
  const l5 = windowCounts(games, 25, {}, 'Under').find(count => count.key === 5);
  assert.deepEqual([l5.hits, l5.decided], [2, 3]);
});

const game = (n, opponent, value, extra = {}) => ({ date: `2026-05-${String(n).padStart(2, '0')}`, opponent, value, season: '2025-26', ...extra });
const games = [
  game(1, 'vs. Celtics', 20), game(2, '@ Knicks', 30), game(3, 'vs. Celtics', 25), game(4, '@ Heat', 10),
  game(5, 'vs. Magic', 25), game(6, '@ Celtics', null), game(7, 'vs. Bulls', 30),
];
const ctx = { nextOpponent: 'Celtics', currentSeason: '2025-26', previousSeason: '2024-25' };

test('a numeric window is the most recent N games', () => {
  assert.deepEqual(selectWindow(games, 3, ctx).map(g => g.date), ['2026-05-05', '2026-05-06', '2026-05-07']);
});

test('head to head keeps games against the next opponent only', () => {
  assert.deepEqual(selectWindow(games, 'h2h', ctx).map(g => g.date), ['2026-05-01', '2026-05-03', '2026-05-06']);
  assert.deepEqual(selectWindow(games, 'h2h', { ...ctx, nextOpponent: null }), []);
});

test('home and away use the is_home flag, falling back to the vs. and @ markers', () => {
  const flagged = [game(1, 'Celtics', 10, { is_home: true }), game(2, 'Knicks', 20, { is_home: false }), game(3, 'vs. Heat', 30), game(4, '@ Magic', 40)];
  assert.deepEqual(selectWindow(flagged, 'home', ctx).map(g => g.date), ['2026-05-01', '2026-05-03']);
  assert.deepEqual(selectWindow(flagged, 'away', ctx).map(g => g.date), ['2026-05-02', '2026-05-04']);
});

test('season windows need a known season and never match games without one', () => {
  const mixed = [game(1, 'vs. A', 1, { season: '2024-25' }), game(2, 'vs. B', 2), game(3, 'vs. C', 3, { season: undefined })];
  assert.deepEqual(selectWindow(mixed, 'previous', ctx).map(g => g.date), ['2026-05-01']);
  assert.deepEqual(selectWindow(mixed, 'current', { ...ctx, currentSeason: undefined }), []);
});

test('counts grade each window over the line: a tie is a push, a missing value is not a game', () => {
  const counts = windowCounts(games, 25, ctx);
  const l5 = counts.find(c => c.key === 5);
  // last five games: 25, 10, 25, (missing), 30 -> four real results
  assert.deepEqual({ n: l5.n, hits: l5.hits, decided: l5.decided, pushes: l5.pushes, rate: l5.rate }, { n: 4, hits: 1, decided: 2, pushes: 2, rate: 50 });
});

test('a window with no games has a null rate, never zero', () => {
  const counts = windowCounts([], 20, ctx);
  assert.ok(counts.every(c => c.n === 0 && c.rate === null));
  assert.equal(counts.length, 8);
});
