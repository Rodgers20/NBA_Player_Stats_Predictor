import test from 'node:test';
import assert from 'node:assert/strict';
import { american, fairOdds, pct, tipText, statusText, lineLabel, windowGames, hitSummary, defaultLine } from './nhl.mjs';
import { boardRows, boardGames, lineOptions, initialLine } from './nhl-props.mjs';

test('american formats signs and nulls', () => {
  assert.equal(american(126), '+126');
  assert.equal(american(-127), '-127');
  assert.equal(american(null), null);
});

test('fairOdds converts a probability to a no-vig price', () => {
  assert.equal(fairOdds(0.6), -150);
  assert.equal(fairOdds(0.4), 150);
  assert.equal(fairOdds(0.5), -100);
  assert.equal(fairOdds(0), null);
  assert.equal(fairOdds(null), null);
});

test('pct, tipText and statusText', () => {
  assert.equal(pct(0.571), '57%');
  assert.equal(pct(null), '—');
  assert.equal(tipText('2026-11-04T00:00Z'), '7:00 PM ET');
  assert.equal(tipText('nope'), 'TBA');
  assert.equal(statusText({ status: 'final', period_type: 'SO' }), 'Final / SO');
  assert.equal(statusText({ status: 'final', period_type: 'REG' }), 'Final');
  assert.equal(statusText({ status: 'live' }), 'Live');
});

test('lineLabel reads anytime for goals and N+ otherwise', () => {
  assert.equal(lineLabel('goal', 0.5), 'Anytime');
  assert.equal(lineLabel('sog', 2.5), '3+');
});

test('windowGames returns the newest N oldest-first; 0 means this season', () => {
  const log = [5, 4, 3, 2, 1].map(n => ({ n }));
  assert.deepEqual(windowGames(log, 2).map(g => g.n), [4, 5]);
  assert.deepEqual(windowGames(log, 0, 3).map(g => g.n), [3, 4, 5]);
  assert.deepEqual(windowGames(log, 10).length, 5);
});

test('hitSummary counts values over the line and skips nulls', () => {
  assert.deepEqual(hitSummary([3, 2, null, 4], 2.5), { n: 3, hits: 2, rate: 2 / 3 });
  assert.equal(hitSummary([], 0.5).rate, null);
});

test('defaultLine prefers 2.5 then 0.5', () => {
  assert.equal(defaultLine([0.5, 1.5, 2.5]), 2.5);
  assert.equal(defaultLine([1.5, 0.5]), 0.5);
  assert.equal(defaultLine([]), null);
});

const player = (id, name, team, opp, game_id, p, extra = {}) => ({
  player_id: id, name, team, opp, home: true, game_id, date: '2026-11-03', start: `2026-11-04T0${game_id % 10}:00Z`,
  markets: { sog: { projection: p * 5, lines: [{ line: 2.5, p_over: p, hit_l10: extra.l10 ?? null }, { line: 3.5, p_over: p / 2, hit_l10: null }] } }, ...extra,
});
const players = [player(1, 'Ann Able', 'TOR', 'UTA', 1, 0.6, { l10: 0.5 }), player(2, 'Bo Béné', 'UTA', 'TOR', 1, 0.8, { l10: 0.9 }), player(3, 'Cy Cole', 'BOS', 'NYR', 2, 0.7, { date: '2026-11-04' })];

test('boardRows sorts by probability and filters by date, game, team and accent-free search', () => {
  assert.deepEqual(boardRows(players, { market: 'sog', line: 2.5 }).map(r => r.player.player_id), [2, 3, 1]);
  assert.deepEqual(boardRows(players, { market: 'sog', line: 2.5, date: '2026-11-03' }).map(r => r.player.player_id), [2, 1]);
  assert.deepEqual(boardRows(players, { market: 'sog', line: 2.5, gameId: '2' }).map(r => r.player.player_id), [3]);
  assert.deepEqual(boardRows(players, { market: 'sog', line: 2.5, team: 'TOR' }).map(r => r.player.player_id), [2, 1]);
  assert.deepEqual(boardRows(players, { market: 'sog', line: 2.5, query: 'bene' }).map(r => r.player.player_id), [2]);
  assert.deepEqual(boardRows(players, { market: 'sog', line: 2.5, sort: 'l10' }).map(r => r.player.player_id), [2, 1, 3]);
});

test('rows need the market and line; options and defaults come from the board', () => {
  assert.equal(boardRows(players, { market: 'goal', line: 0.5 }).length, 0);
  assert.deepEqual(lineOptions(players, 'sog'), [2.5, 3.5]);
  assert.equal(initialLine([2.5, 3.5], 'sog'), 2.5);
  assert.equal(initialLine([1.5, 3.5], 'sog'), 1.5);
  assert.equal(initialLine([], 'goal'), null);
});

test('boardGames lists each game once in start order', () => {
  assert.deepEqual(boardGames(players).map(g => g.id), [1, 2]);
  assert.deepEqual(boardGames(players, '2026-11-04').map(g => g.id), [2]);
});
