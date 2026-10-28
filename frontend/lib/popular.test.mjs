import test from 'node:test';
import assert from 'node:assert/strict';
import { popularBoard, isStarted } from './popular.mjs';

const NOW = Date.parse('2026-10-22T20:00:00Z');
const prop = (over = {}) => ({
  player: 'A', stat: 'PTS', direction: 'Over', line: 10.5, price: -110, ev: 0.05, pick_type: 'pick',
  game_matchup: 'AAA @ BBB', hit_windows: { L10: { hits: 8, total: 10, pct: 80 } }, ...over,
});
const names = rows => rows.map(row => row.player);

test('hottest ranks by last-10 hit rate, then edge, then sample size', () => {
  const rows = [
    prop({ player: 'low', hit_windows: { L10: { hits: 6, total: 10, pct: 60 } } }),
    prop({ player: 'hot-small-edge', ev: 0.01, hit_windows: { L10: { hits: 9, total: 10, pct: 90 } } }),
    prop({ player: 'hot-big-edge', ev: 0.09, hit_windows: { L10: { hits: 9, total: 10, pct: 90 } } }),
  ];
  assert.deepEqual(names(popularBoard(rows, [], { by: 'hit', now: NOW })), ['hot-big-edge', 'hot-small-edge', 'low']);
});

test('best edge keeps only real picks and ranks by expected value', () => {
  const rows = [prop({ player: 'lean', pick_type: 'lean', ev: -0.02 }), prop({ player: 'small', ev: 0.02 }), prop({ player: 'big', ev: 0.1 })];
  assert.deepEqual(names(popularBoard(rows, [], { by: 'edge', now: NOW })), ['big', 'small']);
});

test('unpriced research, missing lines, near-certain prices and thin samples never appear', () => {
  const rows = [
    prop({ player: 'ok' }),
    prop({ player: 'research', pick_type: 'research', price: null }),
    prop({ player: 'no-line', line: null }),
    prop({ player: 'lock', price: -2000 }),
    prop({ player: 'thin', hit_windows: { L10: { hits: 3, total: 3, pct: 100 } } }),
    prop({ player: 'no-window', hit_windows: null }),
  ];
  assert.deepEqual(names(popularBoard(rows, [], { by: 'hit', now: NOW })), ['ok']);
});

test('a game that has started or finished drops out, an unknown game is kept', () => {
  const games = [
    { matchup: 'LIVE @ GAME', status_state: 'in', start_time: '2026-10-22T19:00:00Z' },
    { matchup: 'DONE @ GAME', status_state: 'post', start_time: '2026-10-22T17:00:00Z' },
    { matchup: 'PAST @ TIP', status_state: 'pre', start_time: '2026-10-22T19:59:00Z' },
    { matchup: 'LATER @ TONIGHT', status_state: 'pre', start_time: '2026-10-23T00:00:00Z' },
  ];
  const rows = ['LIVE @ GAME', 'DONE @ GAME', 'PAST @ TIP', 'LATER @ TONIGHT', 'NOT @ LISTED']
    .map(game_matchup => prop({ player: game_matchup, game_matchup }));
  assert.deepEqual(names(popularBoard(rows, games, { by: 'hit', now: NOW })), ['LATER @ TONIGHT', 'NOT @ LISTED']);
});

test('isStarted trusts the state first and the clock second, and never guesses on missing data', () => {
  assert.equal(isStarted({ status_state: 'in' }, NOW), true);
  assert.equal(isStarted({ status_state: 'pre', start_time: '2026-10-22T19:00:00Z' }, NOW), true);
  assert.equal(isStarted({ status_state: 'pre', start_time: '2026-10-22T21:00:00Z' }, NOW), false);
  assert.equal(isStarted({ status_state: 'pre' }, NOW), false);
  assert.equal(isStarted(undefined, NOW), false);
});

test('the board is capped, and the input is left alone', () => {
  const rows = Array.from({ length: 40 }, (_, i) => prop({ player: `p${i}`, ev: i / 100 }));
  assert.equal(popularBoard(rows, [], { by: 'edge', now: NOW }).length, 25);
  assert.equal(popularBoard(rows, [], { by: 'edge', now: NOW, limit: 3 }).length, 3);
  assert.equal(rows[0].player, 'p0');
});
