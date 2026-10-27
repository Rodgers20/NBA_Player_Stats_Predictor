import test from 'node:test';
import assert from 'node:assert/strict';
import { boxScoreHref, boxScoreIndex } from './box-links.mjs';

const teams = { teams: [
  { abbrev: 'BOS', games: [{ id: '401909088', date: '2026-10-28T00:10Z' }, { id: '401909001', date: '2026-10-22T23:30Z' }] },
  { abbrev: 'BKN', games: [{ id: '401909088', date: '2026-10-28T00:10Z' }] },
] };

test('a game-log row links to its box score by team and Eastern game date', () => {
  const index = boxScoreIndex(teams);
  // 00:10 UTC on the 28th is the evening of the 27th in New York, the date the game log shows.
  assert.equal(boxScoreHref(index, { date: '2026-10-27', opponent: 'BOS vs. BKN' }, 'nba'), '/games/box?id=401909088&league=nba');
  assert.equal(boxScoreHref(index, { date: '2026-10-27', opponent: 'BKN @ BOS' }, 'nba'), '/games/box?id=401909088&league=nba');
  assert.equal(boxScoreHref(index, { date: '2026-10-22', opponent: 'BOS @ MIA' }, 'nba'), '/games/box?id=401909001&league=nba');
});

test('games outside the schedule (older seasons, unknown teams) get no link', () => {
  const index = boxScoreIndex(teams);
  assert.equal(boxScoreHref(index, { date: '2025-04-01', opponent: 'BOS vs. CHI' }, 'nba'), null);
  assert.equal(boxScoreHref(index, { date: '2026-10-27', opponent: '' }, 'nba'), null);
  assert.equal(boxScoreHref(boxScoreIndex(undefined), { date: '2026-10-27', opponent: 'BOS vs. BKN' }, 'nba'), null);
});
