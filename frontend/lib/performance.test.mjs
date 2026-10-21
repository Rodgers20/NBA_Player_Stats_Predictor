import test from 'node:test';
import assert from 'node:assert/strict';
import {metric,percentage,calibrationPoints,selectedSeason,seasonLabel} from './performance.mjs';

test('missing metrics never become zero or claimed accuracy',()=>{
  for(const value of [null,undefined,NaN,Infinity]) {assert.equal(metric(value),'—');assert.equal(percentage(value),'—');}
  assert.equal(metric(0),'0.0');assert.equal(percentage(.75),'75.0%');
});
test('calibration excludes malformed or empty bins and orders probabilities',()=>{
  const bins=[{count:12,predicted:.8,actual:.7},{count:20,predicted:.2,actual:.3},{count:0,predicted:.4,actual:.5},{count:4,predicted:2,actual:.3}];
  assert.deepEqual(calibrationPoints(bins).map(b=>b.predicted),[.2,.8]);
});
test('season selection follows league current season when requested archive is missing',()=>{
  const live={current_season:2026,seasons:[{season:2026},{season:2025}]};
  assert.equal(selectedSeason(live,2025).season,2025);
  assert.equal(selectedSeason(live,2024).season,2026);
  assert.equal(seasonLabel(2025,'nba'),'2025–26');assert.equal(seasonLabel(2026,'wnba'),'2026');
});

import {modelContext} from './model-context.mjs';
test('validated model probability and important limitations are transparent',()=>{
  const result=modelContext({winner_probability:.73,model_metadata:{validated:true,context_warning:'Preseason was not validated',injury_scenario_applied:true,injury_adjustment_validated:false},data_quality:{teams:{BOS:{history_older_than_14_days:true}}}});
  assert.equal(result.probability,.73);
  assert.equal(result.warnings.length,3);
  assert.equal(modelContext({winner_probability:.9,model_metadata:{validated:false}}).probability,null);
});
