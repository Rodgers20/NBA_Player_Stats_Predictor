import test from 'node:test';
import assert from 'node:assert/strict';
import { evPerUnit, gameEdges, bettableEdges, EDGE_MIN } from './game-edges.mjs';

const game = (matchup, home, away, homeMl, awayMl, extra = {}) => ({ matchup, home_team: home, away_team: away, home_ml: homeMl, away_ml: awayMl, status_state: 'pre', ...extra });

test('expected value per unit staked from a model chance and an American price', () => {
  assert.equal(Math.round(evPerUnit(0.6, 100) * 1000) / 1000, 0.2);   // 0.6*1 - 0.4
  assert.equal(Math.round(evPerUnit(0.5, -110) * 1000) / 1000, -0.045);
  assert.equal(evPerUnit(0.5, null), null);
  assert.equal(evPerUnit(1, 100), null, 'a certain outcome is not a probability to trust');
  assert.equal(evPerUnit(0, 100), null);
});

test('each side of a game is priced from the model win chance, home and away summing to one', () => {
  const rows = gameEdges([game('BOS @ DET', 'DET', 'BOS', 100, -120)], [{ matchup: 'BOS @ DET', home_win_probability: 0.6 }]);
  const home = rows.find(r => r.side === 'home');
  const away = rows.find(r => r.side === 'away');
  assert.equal(home.team, 'DET');
  assert.ok(Math.abs(home.modelProb + away.modelProb - 1) < 1e-9);
  assert.ok(home.ev > 0.19 && home.ev < 0.21);
  assert.ok(away.ev < 0);
});

test('rows are ranked by edge, best first', () => {
  const rows = gameEdges(
    [game('A @ B', 'B', 'A', 100, -120), game('C @ D', 'D', 'C', -150, 130)],
    [{ matchup: 'A @ B', home_win_probability: 0.6 }, { matchup: 'C @ D', home_win_probability: 0.62 }],
  );
  assert.deepEqual(rows.map(r => r.ev), [...rows.map(r => r.ev)].sort((a, b) => b - a));
});

test('a game with no price, no model chance, or already started is left out', () => {
  const predictions = [{ matchup: 'A @ B', home_win_probability: 0.6 }, { matchup: 'C @ D', home_win_probability: null }, { matchup: 'E @ F', home_win_probability: 0.7 }];
  const rows = gameEdges([
    game('A @ B', 'B', 'A', null, null),
    game('C @ D', 'D', 'C', 100, -120),
    game('E @ F', 'F', 'E', 100, -120, { status_state: 'in' }),
    game('G @ H', 'H', 'G', 100, -120),
  ], predictions);
  assert.deepEqual(rows, []);
});

test('a single missing price drops only that side', () => {
  const rows = gameEdges([game('A @ B', 'B', 'A', 100, null)], [{ matchup: 'A @ B', home_win_probability: 0.6 }]);
  assert.deepEqual(rows.map(r => r.side), ['home']);
});

test('only edges at or above the minimum are bettable, and the minimum is a real margin', () => {
  assert.ok(EDGE_MIN >= 0.02);
  const rows = [{ ev: EDGE_MIN }, { ev: EDGE_MIN - 0.001 }, { ev: -0.1 }];
  assert.deepEqual(bettableEdges(rows).map(r => r.ev), [EDGE_MIN]);
});
