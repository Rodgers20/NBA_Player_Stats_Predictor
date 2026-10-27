import test from 'node:test';
import assert from 'node:assert/strict';
import { importPicks, parsePick, splitPicks } from './pick-import.mjs';

const prop = (player, stat, line, over, under, extra = {}) => ({ player, stat, line, team: 'NYK', game_matchup: 'MIN @ NYK', direction: 'Over',
  model_prob: 0.55, books: [{ book: 'FanDuel', line, over_price: over, under_price: under }], ...extra });
const board = { target_date: '2026-10-27', props: [
  prop('Jalen Brunson', 'PTS', 27.5, -115, -105),
  prop('Anthony Edwards', 'AST', 4.5, 105, -125, { team: 'MIN' }),
  prop('Shai Gilgeous-Alexander', 'PTS+REB+AST', 41.5, -110, -110, { team: 'OKC', game_matchup: 'OKC @ DEN' }),
  prop('Nikola Jokić', 'REB', 12.5, -120, -110, { team: 'DEN', game_matchup: 'OKC @ DEN' }),
  prop('Jalen Williams', 'PTS', 21.5, -110, -110, { team: 'OKC', game_matchup: 'OKC @ DEN' }),
] };
const boards = { nba: board };

test('a post splits into one pick per line, comma or bullet, and drops chatter', () => {
  assert.deepEqual(splitPicks('🔥 3-leg tonight 🔥\n1. Brunson o27.5 pts\n• Ant over 4.5 ast, Jokic O12.5 reb'),
    ['Brunson o27.5 pts', 'Ant over 4.5 ast', 'Jokic O12.5 reb']);
});

test('shorthand becomes direction, line and stat', () => {
  assert.deepEqual(parsePick('Brunson o27.5 pts'), { query: 'brunson', direction: 'Over', line: 27.5, stat: 'PTS', assumedDirection: false });
  assert.deepEqual(parsePick('SGA u 41.5 P+R+A (-110)'), { query: 'sga', direction: 'Under', line: 41.5, stat: 'PTS+REB+AST', assumedDirection: false });
  assert.equal(parsePick('Brunson 30+ points').line, 29.5, '30+ means 30 or more, which is over 29.5');
  assert.equal(parsePick('Jokic 12.5 rebounds').assumedDirection, true);
  assert.equal(parsePick('Ant over 4.5 dimes').stat, 'AST');
  assert.equal(parsePick('let’s ride tonight'), null);
});

test('an exact line at a book is ready to add, with that book’s price', () => {
  const [pick] = importPicks('Brunson o27.5 pts', boards);
  assert.equal(pick.status, 'exact');
  assert.deepEqual([pick.leg.player, pick.leg.direction, pick.leg.line, pick.leg.price, pick.leg.matchup], ['Jalen Brunson', 'Over', 27.5, -115, 'MIN @ NYK']);
  assert.equal(pick.leg.probability, 0.55, 'the model chance carries over only for the model’s own side and line');
  assert.equal(importPicks('Brunson u27.5 pts', boards)[0].leg.probability, null);
});

test('nicknames, initials and accents find the player', () => {
  assert.equal(importPicks('Ant over 4.5 ast', boards)[0].leg.player, 'Anthony Edwards');
  assert.equal(importPicks('SGA o41.5 pra', boards)[0].leg.player, 'Shai Gilgeous-Alexander');
  assert.equal(importPicks('Jokic O12.5 reb', boards)[0].leg.player, 'Nikola Jokić');
});

test('a different line is offered, never substituted', () => {
  const [pick] = importPicks('Brunson over 25.5 points', boards);
  assert.equal(pick.status, 'line');
  assert.deepEqual([pick.leg.line, pick.leg.price, pick.requested], [27.5, -115, 25.5]);
});

test('unknown, ambiguous and missing markets are reported, not guessed', () => {
  assert.equal(importPicks('Wembanyama o3.5 blk', boards)[0].status, 'player');
  const [jalen] = importPicks('Jalen o21.5 pts', boards);
  assert.equal(jalen.status, 'ambiguous');
  assert.deepEqual(jalen.candidates.sort(), ['Jalen Brunson', 'Jalen Williams']);
  assert.equal(importPicks('Brunson o5.5 ast', boards)[0].status, 'market');
});
