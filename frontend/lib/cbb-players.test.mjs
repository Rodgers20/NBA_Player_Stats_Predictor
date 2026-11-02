import test from 'node:test';
import assert from 'node:assert/strict';
import { statValue, boardRows, defaultSeason, seasonLabel, bestFirst, formatStat } from './cbb-players.mjs';

const mk = (id, name, team, pos, line26, line27) => ({ id, name, team, pos, stats: { ...(line26 && { 2026: line26 }), ...(line27 && { 2027: line27 }) } });
const A = { gp: 30, min: 33, pts: 20, reb: 5, ast: 4, stl: 1, blk: 0.5, tov: 3 };
const B = { gp: 3, min: 30, pts: 30, reb: 9, ast: 1, stl: 2, blk: 2, tov: 1 };
const C = { gp: 28, min: 25, pts: 10, reb: 11, ast: 2, stl: 0.5, blk: 2.5, tov: 2 };
const players = [mk('1', 'Ann Able', 'Kansas', 'G', A), mk('2', 'Bo Béné', 'Duke', 'F', B, B), mk('3', 'Cy Cole', 'Kansas', 'C', C)];

test('derived stats add up and a missing line is null', () => {
  assert.equal(statValue(A, 'pra'), 29);
  assert.equal(statValue(A, 'stocks'), 1.5);
  assert.equal(statValue({ pts: 3 }, 'pra'), null);
  assert.equal(statValue(undefined, 'pts'), null);
});

test('sorts best first and ranks the filtered list', () => {
  const rows = boardRows(players, { season: 2026, sort: 'reb', dir: -1 });
  assert.deepEqual(rows.map(r => [r.name, r.rank]), [['Cy Cole', 1], ['Bo Béné', 2], ['Ann Able', 3]]);
});

test('minimum games drops small samples; season without a line drops the player', () => {
  assert.deepEqual(boardRows(players, { season: 2026, minGames: 10 }).map(r => r.id), ['1', '3']);
  assert.deepEqual(boardRows(players, { season: 2027 }).map(r => r.id), ['2']);
});

test('search ignores accents and matches team; position filter matches letters', () => {
  assert.deepEqual(boardRows(players, { season: 2026, query: 'bene' }).map(r => r.id), ['2']);
  assert.deepEqual(boardRows(players, { season: 2026, query: 'kansas' }).map(r => r.id).sort(), ['1', '3']);
  assert.deepEqual(boardRows(players, { season: 2026, position: 'C' }).map(r => r.id), ['3']);
});

test('turnovers rank fewest first; ascending flips the order', () => {
  assert.equal(bestFirst('tov'), 1);
  assert.equal(bestFirst('pts'), -1);
  assert.equal(boardRows(players, { season: 2026, sort: 'tov', dir: 1 })[0].id, '2');
});

test('default season waits for enough real games, then switches to the running one', () => {
  const few = { season: 2027, seasons: [2027, 2026], players };
  assert.equal(defaultSeason(few, 2, 5), 2026);
  assert.equal(defaultSeason({ ...few, players: [players[1], { ...players[1], id: '9' }] }, 2, 3), 2027);
  assert.equal(seasonLabel(2027), '2026-27');
  assert.equal(formatStat(null, 'pts'), '–');
  assert.equal(formatStat(12.345, 'pts'), '12.3');
  assert.equal(formatStat(30, 'gp'), '30');
});
