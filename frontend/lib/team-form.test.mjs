import test from 'node:test';
import assert from 'node:assert/strict';
import { recentScoringGames } from './team-form.mjs';

test('latest scoring games are chronological regardless of provider order', () => {
  const games = Array.from({ length: 25 }, (_, index) => ({ id: String(index + 1), date: `2026-05-${String(index + 1).padStart(2, '0')}T23:00:00Z`, points_for: 80, points_against: 75 }));
  const newestFirst = [...games].reverse();
  const pending = { id: 'pending', date: '2026-05-26T23:00:00Z', points_for: null, points_against: null };
  const expected = games.slice(5).map(g => g.id);
  assert.deepEqual(recentScoringGames([...newestFirst, pending]).map(g => g.id), expected);
  assert.deepEqual(recentScoringGames(games).map(g => g.id), expected);
  assert.equal(newestFirst[0].id, '25', 'input order is not mutated');
});
