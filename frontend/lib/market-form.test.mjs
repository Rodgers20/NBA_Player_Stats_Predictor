import test from 'node:test';
import assert from 'node:assert/strict';
import { marketForm } from './market-form.mjs';
import { gradeSide } from './hit-rate.mjs';

const game = (date, pf, pa) => ({ id: date, date, opponent: 'X', is_home: true, points_for: pf, points_against: pa });
const home = [game('2026-10-10', 110, 107), game('2026-10-12', 100, 102)]; // margins +3, -2
const away = [game('2026-10-11', 120, 100), game('2026-10-13', 90, 100)]; // margins +20, -10

test('moneyline form is each recent margin against zero', () => {
  const form = marketForm({ spread: null, total: null }, home, away);
  assert.deepEqual(form.moneyline.home, { values: [3, -2], line: 0 });
  assert.deepEqual(form.moneyline.away, { values: [20, -10], line: 0 });
});

test('spread form grades a margin against the handicap each team carries today', () => {
  const form = marketForm({ spread: 1.5, total: null }, home, away); // home +1.5, away -1.5
  assert.deepEqual(form.spread.home.values.map(v => gradeSide(v, form.spread.home.line, 'Over')), ['hit', 'miss']);
  assert.deepEqual(form.spread.away.values.map(v => gradeSide(v, form.spread.away.line, 'Over')), ['hit', 'miss']);
});

test('total form is combined points per recent game against the total', () => {
  const form = marketForm({ spread: null, total: 221.5 }, home, away);
  assert.deepEqual(form.total, { home: [217, 202], away: [220, 190], line: 221.5 });
});

test('no line means no spread or total form, never a zero line', () => {
  const form = marketForm({ spread: null, total: null }, home, away);
  assert.equal(form.spread, null);
  assert.equal(form.total, null);
});

test('games without a final score are ignored and only the latest N are used', () => {
  const noisy = [game('2026-10-01', 90, 80), { ...game('2026-10-02', null, null) }, game('2026-10-03', 95, 90)];
  assert.deepEqual(marketForm({ spread: null, total: null }, noisy, [], 1).moneyline.home.values, [5]);
  assert.deepEqual(marketForm({ spread: null, total: null }, noisy, []).moneyline.away.values, []);
});
