import test from 'node:test';
import assert from 'node:assert/strict';
import { recentScoringGames, withPriorSeason } from './team-form.mjs';

const played = (id, date) => ({ id, date, points_for: 100, points_against: 90 });

test('a team short on games this season borrows its latest games from the prior season', () => {
  const current = [{ abbrev: 'BOS', games: [played('c1', '2026-10-20')] }];
  const prior = [{ abbrev: 'BOS', games: [played('p1', '2026-05-01'), played('p2', '2026-05-03')] }];
  const [bos] = withPriorSeason(current, prior, 3);
  assert.deepEqual(bos.games.map(g => g.id), ['p1', 'p2', 'c1']);
  assert.equal(current[0].games.length, 1, 'input is not mutated');
});

test('a team with enough games this season is left alone, and unmatched teams are kept', () => {
  const current = [{ abbrev: 'BOS', games: [played('c1', '2026-10-20'), played('c2', '2026-10-22')] }, { abbrev: 'NEW', games: [] }];
  const prior = [{ abbrev: 'BOS', games: [played('p1', '2026-05-01')] }];
  const result = withPriorSeason(current, prior, 2);
  assert.deepEqual(result[0].games.map(g => g.id), ['c1', 'c2']);
  assert.deepEqual(result[1].games, []);
  assert.deepEqual(withPriorSeason(current, undefined, 2), current);
});

test('latest scoring games are chronological regardless of provider order', () => {
  const games = Array.from({ length: 25 }, (_, index) => ({ id: String(index + 1), date: `2026-05-${String(index + 1).padStart(2, '0')}T23:00:00Z`, points_for: 80, points_against: 75 }));
  const newestFirst = [...games].reverse();
  const pending = { id: 'pending', date: '2026-05-26T23:00:00Z', points_for: null, points_against: null };
  const expected = games.slice(5).map(g => g.id);
  assert.deepEqual(recentScoringGames([...newestFirst, pending]).map(g => g.id), expected);
  assert.deepEqual(recentScoringGames(games).map(g => g.id), expected);
  assert.equal(newestFirst[0].id, '25', 'input order is not mutated');
});
