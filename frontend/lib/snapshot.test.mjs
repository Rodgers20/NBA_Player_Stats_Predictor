import test from 'node:test';
import assert from 'node:assert/strict';
import { playerKey, filterProps, windowChart, seriesChart, filterDefense, filterTrends } from './snapshot.mjs';
test('defense and trend snapshots filter like the live endpoints', () => {
  const defense = { league: 'nba', stat: null, season: null, rankings: [{ stat: 'PTS' }, { stat: 'REB' }] };
  assert.deepEqual(filterDefense(defense, 'reb').rankings, [{ stat: 'REB' }]);
  const trends = { league: 'nba', count: 3, trends: [{ side: 'over', stat: 'PTS' }, { side: 'under', stat: 'PTS' }, { side: 'over', stat: 'REB' }] };
  assert.equal(filterTrends(trends, { side: 'over', stat: 'PTS', limit: 5 }).count, 1);
  assert.equal(filterTrends(trends, { side: 'all' }).count, 3);
});
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
test("a snapshot dated before today is withheld, so yesterday's picks never show as today's", () => {
  const prop = { direction: 'Over', game_matchup: 'A @ B', stat: 'PTS', ev: 3, is_lock: false };
  const data = { target_date: '2026-10-21', game_matchups: ['A @ B'], count: 1, props: [prop] };
  const result = filterProps(data, {}, new Date('2026-10-22T14:00:00Z'));
  assert.equal(result.count, 0); assert.deepEqual(result.props, []);
  assert.deepEqual(result.game_matchups, []); assert.equal(result.status, 'stale');
  assert.equal(result.target_date, '2026-10-21'); assert.match(result.message, /aren.t published yet/);
  assert.equal(data.props.length, 1);
});
test("today's snapshot is served unchanged", () => {
  const prop = { direction: 'Over', game_matchup: 'A @ B', stat: 'PTS', ev: 3, is_lock: false };
  const result = filterProps({ target_date: '2026-10-22', props: [prop] }, {}, new Date('2026-10-22T14:00:00Z'));
  assert.equal(result.count, 1); assert.notEqual(result.status, 'stale');
});
test('a static series keeps the per-game context the game filters need, and omits what is missing', () => {
  const series = { games: [
    { date: '2026-01-01', opponent: 'A vs. B', season: '2025-26', is_home: true, pts: 20, reb: 5, ast: 4, minutes: 31.5, margin: -4, won: false, rest: 0, out: ['Big Man'] },
    { date: '2026-01-03', opponent: 'A @ C', season: '2025-26', is_home: false, pts: 25, reb: 6, ast: 3 },
  ] };
  const [first, second] = seriesChart('Player', series, 'PTS', 2).games;
  assert.deepEqual([first.minutes, first.margin, first.won, first.rest, first.out], [31.5, -4, false, 0, ['Big Man']]);
  assert.equal('minutes' in second, false); assert.equal('out' in second, false);
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

 test('scheduled player snapshots cannot leak another date or league', async () => {
  const { scheduledPlayersSnapshot } = await import('./snapshot.mjs');
  const data = { league: 'nba', game_date: '2026-10-04', players: [{ name: 'Test Player', team: 'NYK' }] };
  assert.deepEqual(scheduledPlayersSnapshot(data, 'nba', '2026-10-04'), data);
  assert.deepEqual(scheduledPlayersSnapshot(data, 'nba', '2026-10-05').players, []);
  assert.deepEqual(scheduledPlayersSnapshot(data, 'wnba', '2026-10-04').players, []);
  assert.deepEqual(scheduledPlayersSnapshot({ ...data, players: [] }, 'nba', '2026-10-04').players, []);
});
