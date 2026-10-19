import { test } from 'node:test';
import assert from 'node:assert/strict';
import { scoreboard, gamePicks, marketProbability, shiftDate } from './game-board.mjs';

test('pregame zeros never replace projections and final scores never fall back to predictions', () => {
  const prediction = {predicted_home_score: 115.5, predicted_away_score: 110.2};
  assert.deepEqual(scoreboard({status_state: 'pre',home_score: 0,away_score: 0}, prediction), {home:115.5,away:110.2,label:'Projected score'});
  assert.deepEqual(scoreboard({status_text:'Final',home_score: 100,away_score: 98}, prediction), {home:100,away:98,label:'Final score'});
  assert.equal(scoreboard({status_state:'in'}, prediction).home, null);
});
test('market gauge requires both valid prices and removes bookmaker margin', () => {
  assert.equal(marketProbability({home_ml:-110,away_ml:-110}), .5);
  assert.equal(marketProbability({home_ml:-200,away_ml:150}), .625);
  assert.equal(marketProbability({home_ml:0,away_ml:150}), null);
  assert.equal(marketProbability({home_ml:-200}), null);
});
test('cover pick flips home line for away and never creates a missing total line', () => {
  assert.deepEqual(gamePicks({spread:-4.5,total:null},{predicted_winner:'BOS',spread_pick:'AWAY',spread_team:'NYK',total_pick:'OVER'}), {winner:'BOS',spread:'NYK +4.5',total:'No line'});
});
test('date ribbon advances across month and year boundaries', () => {
  assert.equal(shiftDate('2026-12-31',1),'2027-01-01');
  assert.equal(shiftDate('2026-10-01',-1),'2026-09-30');
});
test('completed games without a published forecast do not imply pending predictions', () => {
  assert.deepEqual(gamePicks({status_state:'post',spread:-3,total:200}),{winner:'No forecast',spread:'—',total:'—'});
});
