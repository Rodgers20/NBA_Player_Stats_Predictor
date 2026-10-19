import test from 'node:test';
import assert from 'node:assert/strict';
import { averageClv } from './journal-clv.mjs';

test('averages only bets that have a CLV, including zero and negative values', () => {
  const result = averageClv([{ clv: { clv_pct: 2 } }, { clv: { clv_pct: -1 } }, { clv: { clv_pct: 0 } }, { clv: { clv_pct: null } }, {}]);
  assert.deepEqual(result, { count: 3, average: 1 / 3 });
});
test('no close data yields null so the page can show the empty state', () => {
  assert.equal(averageClv([]), null);
  assert.equal(averageClv([{ clv: { clv_pct: null } }, {}]), null);
});
