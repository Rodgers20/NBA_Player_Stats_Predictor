import test from 'node:test';
import assert from 'node:assert/strict';
import { playerKey, filterProps, windowChart, seriesChart } from './snapshot.mjs';
test('snapshot identifiers preserve case punctuation and unicode without path traversal', () => {
  assert.notEqual(playerKey('A B'), playerKey('a-b'));
  assert.match(playerKey('../A’ja'), /^[a-f0-9]+$/);
});
test('static props apply combined filters before counting and paging', () => {
  const prop = { direction:'Under', game_matchup:'A @ B', stat:'PTS', ev:3, is_lock:false };
  const data = {props:[prop,{...prop,ev:4},{...prop,direction:'Over'}]};
  const result = filterProps(data,{direction:'under',stat:'PTS',game:'a @',limit:1});
  assert.equal(result.count,2); assert.equal(result.props[0].ev,4); assert.equal(data.props.length,3);
});
test('static home and away filters match live API location behavior', () => {
  const base = {direction:'Over',game_matchup:'A @ B',stat:'PTS',ev:1,is_lock:false};
  const data = {props:[{...base,is_home_today:true},{...base,is_home_today:false},{...base,is_home_today:null}]};
  assert.equal(filterProps(data,{location:'home'}).count,1);
  assert.equal(filterProps(data,{location:'away'}).count,1);
});
test('chart windows recalculate means and skip missing values', () => {
  const result = windowChart({games:[{value:100},{value:null},{value:4},{value:8}]},3);
  assert.equal(result.avg,6); assert.equal(result.games.length,3);
});
test('compact player series reconstructs base and combined charts', () => {
  const series = { games: [
    {date:'2026-01-01',opponent:'A @ B',pts:12,reb:4,ast:3},
    {date:'2026-01-02',opponent:'A vs. C',pts:8,reb:null,ast:6},
    {date:'2026-01-03',opponent:'A @ D',pts:10,reb:5,ast:2},
  ] };
  const pts = seriesChart('Player', series, 'PTS', 2);
  assert.deepEqual(pts.games.map(row => row.value), [8, 10]);
  assert.equal(pts.avg, 9);
  const pra = seriesChart('Player', series, 'PTS+REB+AST', 3);
  assert.deepEqual(pra.games.map(row => row.value), [19, null, 17]);
  assert.equal(pra.avg, 18);
  assert.equal(pra.l5_avg, 18);
});
