import test from 'node:test';
import assert from 'node:assert/strict';
import { decimalOdds, americanOdds, summarizeSlip, legKey, toggleLeg } from './bet-slip.mjs';
const leg = (i, overrides = {}) => ({league:'nba',date:'2026-10-04',player:`P${i}`,stat:'PTS',matchup:`Game ${i}`,price:-110,probability:0.6,...overrides});
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
