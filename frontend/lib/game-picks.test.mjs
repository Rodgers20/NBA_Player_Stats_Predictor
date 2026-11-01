import test from 'node:test';
import assert from 'node:assert/strict';
import { normalCdf, sigmaFromInterval, modelLeans } from './game-picks.mjs';

test('normalCdf matches known values', () => {
  assert.ok(Math.abs(normalCdf(0) - 0.5) < 1e-6);
  assert.ok(Math.abs(normalCdf(1.2816) - 0.9) < 1e-3);
  assert.ok(Math.abs(normalCdf(-1.2816) - 0.1) < 1e-3);
});

test('sigma comes from the 80% interval and falls back when it is missing', () => {
  assert.ok(Math.abs(sigmaFromInterval([-24.2, 11.7], 'margin') - 14.0) < 0.05);
  assert.equal(sigmaFromInterval(undefined, 'margin'), 12);
  assert.equal(sigmaFromInterval([5, 5], 'total'), 17);
});

// ATL @ NOP: the book has NOP +4.5 / total 239.5; the model has NOP losing by 6.3 and a total of 235.1.
const game = { spread: 4.5, total: 239.5 };
const prediction = { spread: -6.3, total: 235.1, prediction_intervals: { margin: [-24.2, 11.7], total: [211.7, 258.7] } };

test('the model sides with the team that covers: ATL (away) -4.5 when it projects a 6.3-point road win', () => {
  const { spread } = modelLeans(game, prediction);
  assert.equal(spread.side, 'away');
  assert.ok(spread.p > 0.5 && spread.p < 0.7);
});

test('the model takes the under when its total is below the book', () => {
  const { total } = modelLeans(game, prediction);
  assert.equal(total.side, 'under');
  assert.ok(total.p > 0.5);
});

test('home covers when the model margin beats the home handicap', () => {
  const { spread } = modelLeans({ spread: -3, total: 200 }, { spread: 9, total: 210 });
  assert.equal(spread.side, 'home');
});

test('a side within the coin-flip band is no pick, and a missing line gives no pick', () => {
  assert.equal(modelLeans({ spread: -3, total: 200 }, { spread: 3, total: 200.1 }).spread, null);
  assert.equal(modelLeans({ spread: -3, total: 200 }, { spread: 3, total: 200.1 }).total, null);
  assert.deepEqual(modelLeans({ spread: null, total: null }, prediction), { spread: null, total: null });
  assert.deepEqual(modelLeans(game, undefined), { spread: null, total: null });
});
