import test from 'node:test';
import assert from 'node:assert/strict';
import { decimalOdds, americanOdds, summarizeSlip, legKey, toggleLeg, payout, summarizeSingles, slipText, slipStakes, sanitizePersistedSlip, encodeSlip, decodeSlip, MAX_LEGS } from './bet-slip.mjs';
const leg = (i, overrides = {}) => ({league:'nba',date:'2026-10-04',player:`P${i}`,stat:'PTS',direction:'Over',line:20.5,matchup:`Game ${i}`,price:-110,probability:0.6,...overrides});
test('five -110 legs multiply decimal returns and independent model probabilities', () => {
  const result = summarizeSlip(Array.from({length:5}, (_,i)=>leg(i)));
  assert.ok(Math.abs(result.decimal - (21/11)**5) < 1e-10);
  assert.ok(Math.abs(result.probability - .6**5) < 1e-10);
  assert.equal(result.american,2436);
  assert.equal(result.correlated,false);
});
test('odds conversions handle underdogs, favorites and invalid prices', () => {
  assert.equal(decimalOdds(150),2.5); assert.equal(decimalOdds(-200),1.5);
  assert.equal(americanOdds(1.5),-200); assert.equal(americanOdds(2.5),150);
  for (const value of [0,50,-50,NaN,Infinity,null]) assert.equal(decimalOdds(value),null);
});
test('missing probability is never substituted with historical frequency or implied odds', () => {
  assert.equal(summarizeSlip([leg(1,{probability:null})]).probability,null);
  assert.equal(summarizeSlip([]).decimal,null);
  assert.equal(summarizeSlip([leg(1,{price:0})]).decimal,null);
});
test('same-game correlation is identified across different players', () => {
  assert.equal(summarizeSlip([leg(1),leg(2,{matchup:'Game 1'})]).correlated,true);
});
test('market identity prevents duplicate or opposite-side legs while separating dates/leagues', () => {
  assert.equal(legKey(leg(1)),legKey(leg(1,{direction:'Under',line:25.5})));
  assert.notEqual(legKey(leg(1)),legKey(leg(1,{league:'wnba'})));
  assert.notEqual(legKey(leg(1)),legKey(leg(1,{date:'2026-10-05'})));
});

test('adding, removing and replacing a selection leaves only one side of a market', () => {
  const over = leg(1,{direction:'Over',line:20.5});
  const under = leg(1,{direction:'Under',line:20.5});
  const added = toggleLeg([],over);
  assert.deepEqual(added,[over]);
  assert.deepEqual(toggleLeg(added,over),[]);
  assert.deepEqual(toggleLeg(added,under),[under]);
  assert.deepEqual(toggleLeg(added,{...over,line:21.5}),[{...over,line:21.5}]);
});

test('payout includes stake and needs a valid stake and price', () => {
  assert.equal(payout(10, 150), 25); assert.equal(payout(0, 150), null); assert.equal(payout(10, null), null);
});
test('singles total each bet separately and need every price', () => {
  const r = summarizeSingles([leg(1,{price:100}), leg(2,{price:200})], 10);
  assert.deepEqual(r, { bets: 2, totalStake: 20, payout: 50 });
  assert.equal(summarizeSingles([leg(1), leg(2,{price:0})], 10).payout, null);
});
test('slip text names mode, odds and every leg', () => {
  const t = slipText([leg(1,{price:100}), leg(2,{price:100})], 'parlay', 5);
  assert.match(t, /^2-leg parlay \+300, \$5\n/); assert.equal(t.split('\n').length, 3);
  assert.match(slipText([leg(1), leg(2)], 'singles', 5), /^2 singles, \$5 each/);
});
test('share link round-trips including non-ASCII names', () => {
  const legs = [leg(1,{player:"Nikola Jokić"}), leg(2,{probability:null})];
  const back = decodeSlip(encodeSlip(legs, 'singles', 7.5));
  assert.equal(back.mode, 'singles'); assert.equal(back.stake, 7.5);
  assert.deepEqual(back.legs.map(l => [l.player, l.price, l.probability]), [["Nikola Jokić", -110, 0.6], ["P2", -110, null]]);
});
test('decode rejects garbage, drops bad legs, caps leg count', () => {
  for (const bad of [null, '', 'not-base64!!', encodeSlip([], 'parlay', null).slice(2)]) assert.equal(decodeSlip(bad), null);
  const many = Array.from({ length: MAX_LEGS + 5 }, (_, i) => leg(i));
  assert.equal(decodeSlip(encodeSlip(many, 'parlay', null)).legs.length, MAX_LEGS);
  assert.equal(decodeSlip(encodeSlip([leg(1,{price:0})], 'parlay', null)).legs.length, 0);
});
const bank = { units: 100, unitSize: 10 }; // $1000
test('suggested stakes: per leg, singles sum, parlay on the combined price, zero without edge', () => {
  const a = leg(1, { price: 100, probability: 0.55 }), b = leg(2, { price: 100, probability: 0.45 });
  const singles = slipStakes([a, b], 'singles', bank);
  assert.deepEqual(singles.perLeg, [25, 0]); assert.equal(singles.total, 25);
  const pair = [a, leg(3, { price: 100, probability: 0.55 })]; // parlay +300, chance 0.3025
  assert.ok(slipStakes(pair, 'parlay', bank).total > 0);
  assert.equal(slipStakes(pair, 'parlay', bank, 0.2).total, 0); // a correlation-adjusted chance below 25% removes the edge
});
test('persisted slips from before bankroll/team still load; junk is dropped', () => {
  const old = sanitizePersistedSlip({ legs: [leg(1), { nope: 1 }], mode: 'singles', stake: 5 });
  assert.equal(old.legs.length, 1); assert.equal(old.mode, 'singles'); assert.equal(old.stake, 5);
  assert.deepEqual(old.bankroll, { units: 100, unitSize: 10 });
  const bad = sanitizePersistedSlip({ legs: 'x', mode: 'zzz', stake: -3, bankroll: { units: -1, unitSize: 'a' } });
  assert.deepEqual([bad.legs.length, bad.mode, bad.stake], [0, 'parlay', null]);
  assert.deepEqual(sanitizePersistedSlip(null).bankroll, { units: 100, unitSize: 10 });
  assert.deepEqual(sanitizePersistedSlip({ bankroll: { units: 50, unitSize: 20 } }).bankroll, { units: 50, unitSize: 20 });
});

test('sanitizePersistedSlip drops hand-edited legs that would crash rendering', () => {
  const bad = [{ ...leg(1), league: 5 }, { ...leg(2), matchup: null }, { ...leg(3), price: 'x' }, { ...leg(4), price: NaN }];
  assert.deepEqual(sanitizePersistedSlip({ legs: [...bad, leg(5), leg(6, { price: null })] }).legs.map(l => l.player), ['P5', 'P6']);
});
