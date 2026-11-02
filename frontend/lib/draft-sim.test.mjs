import test from 'node:test';
import assert from 'node:assert/strict';
import { rng, projectOrder, drawLottery, pickOrigins, drawBoard, simulate, monteCarlo, PICKS } from './draft-sim.mjs';

const teams = Array.from({ length: 30 }, (_, i) => ({ slug: `t${i}`, w: 3, l: 3, power: (i - 15) / 3 }));   // t0 weakest
const board = Array.from({ length: 70 }, (_, i) => ({ rank: i + 1, name: `P${i + 1}` }));
const balls = [2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1];
const owners = Object.fromEntries(teams.map(t => [t.slug, { r1: t.slug, r2: t.slug }]));
owners.t1 = { r1: 't9', r2: 't1' };                       // t1 traded its first-round pick to t9
const data = { teams, board, owners, lottery: { balls, restricted: { t3: 5 } } };

test('same seed, same draft', () => {
  assert.deepEqual(simulate(data, rng(5)), simulate(data, rng(5)));
});

test('every team has one pick per round and picks are numbered 1-60', () => {
  const d = simulate(data, rng(3));
  assert.equal(d.length, PICKS);
  assert.deepEqual(d.map(p => p.pick), Array.from({ length: PICKS }, (_, i) => i + 1));
  for (const half of [d.slice(0, 30), d.slice(30)]) assert.equal(new Set(half.map(p => p.origin)).size, 30);
});

test('a team barred from the top 5 never lands there; the worst three always land in the top 12', () => {
  const rand = rng(11);
  for (let i = 0; i < 400; i++) {
    const order = projectOrder(teams, rand);
    const lot = drawLottery(order, balls, { t3: 5 }, rand);
    assert.ok(!lot.slice(0, 5).includes('t3'));
    for (const worst of order.slice(0, 3)) assert.ok(lot.indexOf(worst) < 12);
    assert.equal(new Set(lot).size, 16);
  }
});

test('non-lottery picks follow record, and round two runs worst to best', () => {
  const order = teams.map(t => t.slug);
  const origins = pickOrigins(order, order.slice(0, 16).reverse());
  assert.deepEqual(origins.slice(16, 30), order.slice(16));
  assert.deepEqual(origins.slice(30), order);
});

test('traded picks report the new owner', () => {
  const d = simulate(data, rng(2));
  const t1 = d.find(p => p.origin === 't1' && p.pick <= 30);
  assert.equal(t1.owner, 't9');
});

test('a higher-ranked prospect is drafted earlier on average', () => {
  const rand = rng(9);
  let first = 0, tenth = 0;
  const n = 500;
  for (let i = 0; i < n; i++) {
    const order = drawBoard(board, rand);
    first += order.indexOf(0);
    tenth += order.indexOf(9);
  }
  assert.ok(first / n < tenth / n);
});

test('monte carlo odds sum to one per pick and favour the board leader at pick 1', () => {
  const mc = monteCarlo(data, 600, 4);
  for (const k of [0, 14, 44]) {
    assert.ok(Math.abs(mc.owners[k].reduce((s, x) => s + x.p, 0) - 1) < 1e-9);
    assert.ok(Math.abs(mc.players[k].reduce((s, x) => s + x.p, 0) - 1) < 1e-9);
  }
  assert.equal(mc.players[0][0].key, '0');
  assert.ok(mc.profile[0].top3 > mc.profile[30].top3);
  assert.ok(mc.profile[0].expected < mc.profile[30].expected);
});
