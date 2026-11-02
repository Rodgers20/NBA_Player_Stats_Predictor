import test from 'node:test';
import assert from 'node:assert/strict';
import { parseGamelog, windowAverages } from './cbb-gamelog.mjs';

const names = ['minutes', 'fieldGoalsMade-fieldGoalsAttempted', 'fieldGoalPct', 'threePointFieldGoalsMade-threePointFieldGoalsAttempted', 'threePointFieldGoalPct', 'freeThrowsMade-freeThrowsAttempted', 'freeThrowPct', 'totalRebounds', 'assists', 'blocks', 'steals', 'fouls', 'turnovers', 'points'];
const json = {
  names,
  events: {
    a: { gameDate: '2026-03-19T23:35:00.000+00:00', atVs: 'vs', gameResult: 'L', score: '79-71', opponent: { abbreviation: 'TEX' } },
    b: { gameDate: '2026-03-10T23:00:00.000+00:00', atVs: '@', gameResult: 'W', score: '80-70', opponent: { abbreviation: 'UTAH' } },
  },
  seasonTypes: [
    { categories: [{ events: [{ eventId: 'a', stats: ['40', '11-25', '44.0', '1-7', '14.3', '12-12', '100.0', '10', '1', '0', '0', '1', '5', '35'] }] }] },
    { categories: [{ events: [{ eventId: 'b', stats: ['30', '5-10', '50', '2-4', '50', '0-0', '0', '4', '3', '1', '2', '2', '1', '12'] }, { eventId: 'a', stats: [] }] }] },
  ],
};

test('parses stats by name, newest first, each game once', () => {
  const games = parseGamelog(json);
  assert.deepEqual(games.map(g => g.id), ['a', 'b']);
  assert.deepEqual([games[0].pts, games[0].reb, games[0].ast, games[0].tov, games[0].fgm, games[0].fga, games[0].tpm], [35, 10, 1, 5, 11, 25, 1]);
  assert.equal(games[0].home, true);
  assert.equal(games[1].home, false);
  assert.equal(games[1].result, 'W');
});

test('an empty or malformed log gives no games rather than throwing', () => {
  assert.deepEqual(parseGamelog({}), []);
  assert.deepEqual(parseGamelog(null), []);
});

test('window averages cover the newest n games', () => {
  const games = parseGamelog(json);
  assert.equal(windowAverages(games, 1).pts, 35);
  assert.equal(windowAverages(games, 0).pts, 23.5);
  assert.equal(windowAverages([], 5).pts, null);
});
