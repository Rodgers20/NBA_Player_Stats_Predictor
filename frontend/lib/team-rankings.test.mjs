import test from 'node:test';
import assert from 'node:assert/strict';
import { offenseVsDefense } from './team-rankings.mjs';

const metric = (label, side, rank, display = '1.0') => ({ label, side, rank, of: 30, display });
const profiles = {
  AAA: { metrics: [metric('Scoring', 'offense', 5, '120.0'), metric('Rebounding', 'offense', 20, '43.0'), metric('Ball security', 'offense', 10, '13.5'), metric('Scoring defense', 'defense', 8, '110.0')] },
  BBB: { metrics: [metric('Scoring defense', 'defense', 26, '118.0'), metric('Defensive rebounding', 'defense', 12, '40.0'), metric('Forcing turnovers', 'defense', 14, '13.0'), metric('Scoring', 'offense', 9, '116.0')] },
};

test('each offense stat is paired with the defense stat that answers it', () => {
  const rows = offenseVsDefense(profiles, 'AAA', 'BBB');
  assert.deepEqual(rows.map(row => [row.label, row.defenseLabel]), [
    ['Scoring', 'Scoring defense'], ['Rebounding', 'Defensive rebounding'], ['Ball security', 'Forcing turnovers'],
  ]);
  assert.deepEqual(rows[0].offense, { display: '120.0', rank: 5, of: 30 });
  assert.deepEqual(rows[0].defense, { display: '118.0', rank: 26, of: 30 });
});

test('a pairing a team has no data for is left out rather than guessed', () => {
  assert.equal(offenseVsDefense(profiles, 'AAA', 'BBB').some(row => row.label === '3-point shooting'), false);
});

test('the edge goes to whichever side ranks clearly higher, and a small gap is even', () => {
  const [scoring, rebounding, security] = offenseVsDefense(profiles, 'AAA', 'BBB');
  assert.equal(scoring.edge, 'offense');      // offense 5th against a 26th-ranked defense
  assert.equal(rebounding.edge, 'defense');   // offense 20th against a 12th-ranked defense: an 8 place gap
  assert.equal(security.edge, 'even');        // 10th against 14th
});

test('swapping the teams reads the other offense against the first defense', () => {
  const rows = offenseVsDefense(profiles, 'BBB', 'AAA');
  assert.deepEqual(rows.map(row => row.label), ['Scoring']);
  assert.equal(rows[0].offense.rank, 9); assert.equal(rows[0].defense.rank, 8);
});

test('missing profiles give no rows', () => {
  assert.deepEqual(offenseVsDefense(undefined, 'AAA', 'BBB'), []);
  assert.deepEqual(offenseVsDefense(profiles, 'AAA', 'ZZZ'), []);
});
