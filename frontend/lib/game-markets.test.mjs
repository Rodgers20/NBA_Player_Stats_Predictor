import test from 'node:test';
import assert from 'node:assert/strict';
import { fmtPrice, fmtLine, gameMarkets } from './game-markets.mjs';

const played = (id, pf, pa) => ({ id, date: `2026-05-${id}`, points_for: pf, points_against: pa });
const teams = { away: [played('01', 110, 100), played('02', 90, 100)], home: [played('03', 100, 95)] };

test('prices and lines carry an explicit sign and never show zero for missing', () => {
  assert.equal(fmtPrice(120), '+120');
  assert.equal(fmtPrice(-110), '-110');
  assert.equal(fmtPrice(null), '—');
  assert.equal(fmtPrice(undefined), '—');
  assert.equal(fmtLine(1.5), '+1.5');
  assert.equal(fmtLine(-3), '-3');
  assert.equal(fmtLine(0), 'PK');
});

test('a game with no lines has no markets, so nothing reads as zero or NaN', () => {
  const markets = gameMarkets({ spread: null, total: null, away_ml: null, home_ml: null }, 'BOS', 'DET', teams.away, teams.home);
  assert.deepEqual([markets.moneyline, markets.spread, markets.total], [null, null, null]);
  assert.equal(markets.totalTitle, 'Total');
});

test('money line shows each price, with a dash when only one side is priced', () => {
  const { moneyline } = gameMarkets({ spread: null, total: null, away_ml: -120, home_ml: null }, 'BOS', 'DET', teams.away, teams.home);
  assert.deepEqual(moneyline.map(side => [side.label, side.value]), [['BOS', '-120'], ['DET', '—']]);
});

test('spread labels each team with the handicap it carries today', () => {
  const { spread } = gameMarkets({ spread: 1.5, total: null, away_ml: null, home_ml: null }, 'BOS', 'DET', teams.away, teams.home);
  assert.deepEqual(spread.map(side => [side.label, side.value]), [['BOS', '-1.5'], ['DET', '+1.5']]);
  assert.equal(spread[0].form.values.length, 2);
  assert.equal(spread[1].form.values.length, 1);
});

test('captions name the real window and never claim a historical spread record', () => {
  const m = gameMarkets({ spread: 1.5, total: 221.5, away_ml: -120, home_ml: 100 }, 'BOS', 'DET', teams.away, teams.home);
  assert.equal(m.moneyline[0].form.caption, 'L2 wins');
  assert.equal(m.moneyline[1].form.caption, 'L1 wins');
  assert.equal(m.spread[0].form.caption, 'L2 would cover');
  assert.equal(m.total[1].form.caption, 'L1 over');
});

test('total is titled with its line and carries each team combined points', () => {
  const { total, totalTitle } = gameMarkets({ spread: null, total: 221.5, away_ml: null, home_ml: null }, 'BOS', 'DET', teams.away, teams.home);
  assert.equal(totalTitle, 'Total 221.5');
  assert.deepEqual(total[0].form.values, [210, 190]);
  assert.equal(total[0].form.line, 221.5);
});
