import test from 'node:test';
import assert from 'node:assert/strict';
import { streakInsight, scopeLabel, summarize, gameInsights } from './insights.mjs';

test('an under pick reads as staying under the line', () => {
  const text = streakInsight({ player: 'A Player', stat: 'AST', line: 6.5, games: games(3, 2, 4, 5, 1, 4), side: 'Under' });
  assert.equal(text, 'A Player has stayed under 6.5 AST in 6 straight games (3.2 AST/game average).');
});

const pick = (over = {}) => ({
  player: 'P', stat: 'PTS', stat_label: 'Points', line: 20.5, direction: 'Over', price: -110, pick_type: 'pick',
  l10_values: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], ...over, // newest first
});

test('game insights rank straights above near-misses, using the real stat name', () => {
  const rows = gameInsights([
    pick({ player: 'near', l10_values: [10, 30, 31, 32, 33, 34, 35, 36, 37, 38] }),   // 9 of 10, newest under
    pick({ player: 'run', l10_values: [30, 31, 32, 33, 34, 35, 36, 37, 10, 12] }),     // 8 straight
  ]);
  assert.deepEqual(rows.map(row => row.prop.player), ['run', 'near']);
  assert.match(rows[0].text, /^run has exceeded 20\.5 Points in 8 straight games/);
});

test('game insights skip unpriced, near-certain, short-history and unremarkable picks', () => {
  const rows = gameInsights([
    pick({ player: 'ok' }),
    pick({ player: 'research', pick_type: 'research', price: null }),
    pick({ player: 'lock', price: -2000 }),
    pick({ player: 'short', l10_values: [30, 31, 32] }),
    pick({ player: 'meh', l10_values: [30, 10, 31, 12, 32, 11, 33, 10, 34, 9] }),
  ]);
  assert.deepEqual(rows.map(row => row.prop.player), ['ok']);
});

test('game insights are capped and leave the input alone', () => {
  const props = Array.from({ length: 30 }, (_, i) => pick({ player: `p${i}` }));
  assert.equal(gameInsights(props).length, 12);
  assert.equal(gameInsights(props, { limit: 3 }).length, 3);
  assert.equal(props[0].player, 'p0');
});
import { DEFAULT_GAME_FILTERS } from './game-filters.mjs';

const games = (...values) => values.map(value => ({ value }));
const base = { player: 'A Player', stat: 'PTS', line: 20.5 };

test('a current run of games over the line reads as a straight streak', () => {
  const text = streakInsight({ ...base, games: games(10, 22, 25, 30, 28, 31) });
  assert.equal(text, 'A Player has exceeded 20.5 PTS in 5 straight games (27.2 PTS/game average).');
});

test('a game on the line is a push and ends the streak', () => {
  const text = streakInsight({ ...base, games: games(30, 30, 20.5, 25, 26, 27, 28, 29) });
  assert.equal(text, 'A Player has exceeded 20.5 PTS in 5 straight games (27.0 PTS/game average).');
});

test('a streak shorter than five games is not worth a sentence on its own', () => {
  assert.equal(streakInsight({ ...base, games: games(10, 12, 9, 8, 30, 31, 32, 33) }), null);
});

test('a high hit rate without a streak reads as hits out of the last games', () => {
  const mixed = [30, 31, 32, 33, 34, 35, 36, 10, 37, 12]; // 8 of 10, the newest is under
  const run = [30, 31, 10, 32, 33, 34, 35, 36, 37, 38];   // the newest seven are over
  assert.match(streakInsight({ ...base, games: games(...mixed) }), /in 8 of the last 10 games/);
  assert.match(streakInsight({ ...base, games: games(...run) }), /in 7 straight games/);
});

test('a middling hit rate or a small sample says nothing', () => {
  assert.equal(streakInsight({ ...base, games: games(30, 10, 31, 12, 32, 11, 33, 10, 34, 9) }), null);
  assert.equal(streakInsight({ ...base, games: games(30, 31, 32, 10) }), null);
});

test('games with no value are not games', () => {
  const text = streakInsight({ ...base, games: [{ value: null }, ...games(30, 31, 32, 33, 34)] });
  assert.match(text, /in 5 straight games/);
});

test('the scope of any active filters is part of the sentence', () => {
  const text = streakInsight({ ...base, games: games(30, 31, 32, 33, 34), scope: ' at home' });
  assert.equal(text, 'A Player has exceeded 20.5 PTS in 5 straight games at home (32.0 PTS/game average).');
});

test('scope labels describe the filters in plain words', () => {
  assert.equal(scopeLabel(DEFAULT_GAME_FILTERS), '');
  assert.equal(scopeLabel({ ...DEFAULT_GAME_FILTERS, homeOnly: true }), ' at home');
  assert.equal(scopeLabel({ ...DEFAULT_GAME_FILTERS, homeOnly: true, withoutPlayer: 'Big Man' }), ' at home without Big Man');
  assert.equal(scopeLabel({ ...DEFAULT_GAME_FILTERS, withPlayer: 'Wing', result: 'W', rest: '0' }), ' with Wing in wins on a back-to-back');
});

test('summarize gives the mean and median, skipping missing values', () => {
  assert.deepEqual(summarize([10, 20, null, 60]), { mean: 30, median: 20 });
  assert.deepEqual(summarize([4, 8]), { mean: 6, median: 6 });
  assert.equal(summarize([]), null);
  assert.equal(summarize([null, undefined]), null);
});
