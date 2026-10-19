import test from 'node:test';
import assert from 'node:assert/strict';
import { kellyFraction, suggestedStake, bankrollAmount, sanitizeBankroll, DEFAULT_BANKROLL } from './bankroll.mjs';

test('default quarter Kelly at +100 and 55% is a quarter of full Kelly (10%) and the 5% cap binds above it', () => {
  assert.ok(Math.abs(kellyFraction(0.55, 100) - 0.025) < 1e-12);
  assert.equal(kellyFraction(0.7, 100), 0.05); // quarter of 40% = 10%, capped to 5%
});
test('full Kelly matches the closed form and the fraction and cap are configurable', () => {
  const b = 100 / 110, p = 0.56, full = (b * p - (1 - p)) / b;
  assert.ok(Math.abs(kellyFraction(p, -110, { fraction: 1, cap: 1 }) - full) < 1e-12);
  assert.ok(Math.abs(kellyFraction(p, -110, { fraction: 0.5, cap: 1 }) - full / 2) < 1e-12);
  assert.equal(kellyFraction(0.9, 200, { fraction: 1, cap: 0.02 }), 0.02);
});
test('no stake without a positive edge or with invalid inputs', () => {
  assert.equal(kellyFraction(0.5, -110), 0);       // below break-even
  assert.equal(kellyFraction(1 / 2, 100), 0);      // exactly zero edge
  for (const [p, price] of [[null, -110], [0, -110], [1, -110], [NaN, -110], [0.6, 0], [0.6, 50], [0.6, null]]) {
    assert.equal(kellyFraction(p, price), 0);
  }
});
test('suggested stake is in dollars from units times unit size', () => {
  assert.equal(bankrollAmount({ units: 100, unitSize: 10 }), 1000);
  assert.equal(suggestedStake(0.6, 100, { units: 100, unitSize: 10 }), 50);
  assert.equal(suggestedStake(0.5, -110, { units: 100, unitSize: 10 }), 0);
  assert.equal(suggestedStake(0.6, 100, { units: 0, unitSize: 10 }), 0);
});
test('stored bankroll values are sanitized', () => {
  assert.deepEqual(sanitizeBankroll({ units: '50', unitSize: '2.5' }), { units: 50, unitSize: 2.5 });
  for (const bad of [null, {}, { units: -1, unitSize: 5 }, { units: 'x', unitSize: 5 }]) assert.deepEqual(sanitizeBankroll(bad), DEFAULT_BANKROLL);
});
