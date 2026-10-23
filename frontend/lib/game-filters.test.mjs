import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_GAME_FILTERS, MARGIN_CAP, applyGameFilters, activeFilterCount, teammateOptions, histogram, minutesRange, opponentOptions,
} from './game-filters.mjs';

const game = (over = {}) => ({ date: '2026-01-01', is_home: true, rest: 1, out: [], margin: 5, won: true, minutes: 30, ...over });
const filter = (over = {}) => ({ ...DEFAULT_GAME_FILTERS, ...over });
const dates = games => games.map(g => g.date);

test('the default filters keep every game, including ones with unknown context', () => {
  const games = [game(), game({ rest: undefined, margin: undefined, won: undefined, minutes: undefined })];
  assert.equal(applyGameFilters(games, DEFAULT_GAME_FILTERS).length, 2);
  assert.equal(activeFilterCount(DEFAULT_GAME_FILTERS), 0);
});

test('home only drops away games and games with no venue', () => {
  const games = [game({ date: 'h' }), game({ date: 'a', is_home: false }), game({ date: 'u', is_home: null })];
  assert.deepEqual(dates(applyGameFilters(games, filter({ homeOnly: true }))), ['h']);
});

test('away only, opponent and season narrow the games, and each counts as an active filter', () => {
  const games = [
    game({ date: 'h', opponent: 'CHI vs. BOS', season: '2025-26' }),
    game({ date: 'a', is_home: false, opponent: 'CHI @ BOS', season: '2024-25' }),
    game({ date: 'n', is_home: false, opponent: 'NYK', season: '2025-26' }),
  ];
  assert.deepEqual(dates(applyGameFilters(games, filter({ awayOnly: true }))), ['a', 'n']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ opponent: 'BOS' }))), ['h', 'a']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ season: '2025-26' }))), ['h', 'n']);
  assert.equal(activeFilterCount(filter({ awayOnly: true, opponent: 'BOS', season: '2025-26' })), 3);
  assert.deepEqual(opponentOptions(games), ['BOS', 'NYK']);
});

test('days of rest match exactly, with 3+ open ended, and unknown rest never matches', () => {
  const games = [game({ date: '0', rest: 0 }), game({ date: '2', rest: 2 }), game({ date: '5', rest: 5 }), game({ date: '?', rest: undefined })];
  assert.deepEqual(dates(applyGameFilters(games, filter({ rest: '0' }))), ['0']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ rest: '3+' }))), ['5']);
});

test('with and without a teammate split the games on who sat out', () => {
  const games = [game({ date: 'in' }), game({ date: 'out', out: ['Big Man', 'Wing'] })];
  assert.deepEqual(dates(applyGameFilters(games, filter({ withPlayer: 'Big Man' }))), ['in']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ withoutPlayer: 'Big Man' }))), ['out']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ withoutPlayer: 'Nobody' }))), []);
});

test('margin is the size of the result either way, and the top of the range is open ended', () => {
  const games = [game({ date: 'close', margin: 3 }), game({ date: 'loss10', margin: -10 }), game({ date: 'blowout', margin: 30 }), game({ date: '?', margin: undefined })];
  assert.deepEqual(dates(applyGameFilters(games, filter({ margin: [0, 5] }))), ['close']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ margin: [8, 12] }))), ['loss10']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ margin: [10, MARGIN_CAP] }))), ['loss10', 'blowout']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ margin: [0, MARGIN_CAP] }))), ['close', 'loss10', 'blowout', '?']);
});

test('win or loss filters only on a known result', () => {
  const games = [game({ date: 'w', won: true }), game({ date: 'l', won: false }), game({ date: '?', won: undefined })];
  assert.deepEqual(dates(applyGameFilters(games, filter({ result: 'W' }))), ['w']);
  assert.deepEqual(dates(applyGameFilters(games, filter({ result: 'L' }))), ['l']);
});

test('a minutes range is inclusive and drops games with no minutes', () => {
  const games = [game({ date: '20', minutes: 20 }), game({ date: '30', minutes: 30 }), game({ date: '40', minutes: 40 }), game({ date: '?', minutes: undefined })];
  assert.deepEqual(dates(applyGameFilters(games, filter({ minutes: [25, 40] }))), ['30', '40']);
});

test('filters combine, and the active count says how many are on', () => {
  const games = [game({ date: 'keep', rest: 0 }), game({ date: 'away', rest: 0, is_home: false })];
  const combined = filter({ homeOnly: true, rest: '0', margin: [0, 10] });
  assert.deepEqual(dates(applyGameFilters(games, combined)), ['keep']);
  assert.equal(activeFilterCount(combined), 3);
  assert.equal(activeFilterCount(filter({ withPlayer: 'A', withoutPlayer: 'B', result: 'W', minutes: [10, 20] })), 4);
});

test('teammate options are the players who actually sat out, most often first', () => {
  const games = [game({ out: ['B', 'A'] }), game({ out: ['A'] }), game({ out: ['C', 'A'] }), game()];
  assert.deepEqual(teammateOptions(games), ['A', 'B', 'C']);
  assert.deepEqual(teammateOptions([game()]), []);
});

test('a histogram counts values into even bins and clamps the edges', () => {
  assert.deepEqual(histogram([0, 1, 4, 5, 9, 10, 25, null], 0, 10, 2), [3, 4]);
  assert.deepEqual(histogram([], 0, 10, 3), [0, 0, 0]);
});

test('the minutes range follows the games, with a sensible default when there are none', () => {
  assert.deepEqual(minutesRange([game({ minutes: 18.4 }), game({ minutes: 36.2 }), game({ minutes: undefined })]), [18, 37]);
  assert.deepEqual(minutesRange([]), [0, 48]);
});
