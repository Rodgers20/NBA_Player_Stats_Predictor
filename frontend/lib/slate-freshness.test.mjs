import test from 'node:test';
import assert from 'node:assert/strict';
import { todayEt, isPastSlate } from './slate-freshness.mjs';

test('todayEt uses the Eastern calendar day, not UTC', () => {
  // 03:30 UTC on Oct 22 is still the evening of Oct 21 in New York.
  assert.equal(todayEt(new Date('2026-10-22T03:30:00Z')), '2026-10-21');
  assert.equal(todayEt(new Date('2026-10-22T14:00:00Z')), '2026-10-22');
});

test('a slate dated before today (ET) is past', () => {
  assert.equal(isPastSlate('2026-10-21', new Date('2026-10-22T14:00:00Z')), true);
});

test('late-night games stay current until Eastern midnight', () => {
  assert.equal(isPastSlate('2026-10-21', new Date('2026-10-22T03:30:00Z')), false);
});

test("today's and tomorrow's slates are not past", () => {
  const now = new Date('2026-10-22T14:00:00Z');
  assert.equal(isPastSlate('2026-10-22', now), false);
  assert.equal(isPastSlate('2026-10-23', now), false);
});

test('a missing or malformed date proves nothing, so it is not past', () => {
  const now = new Date('2026-10-22T14:00:00Z');
  for (const value of [null, undefined, '', 'tomorrow', '2026-13-45']) {
    assert.equal(isPastSlate(value, now), false);
  }
});
