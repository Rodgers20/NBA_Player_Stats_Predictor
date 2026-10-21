import test from 'node:test';
import assert from 'node:assert/strict';
import { gradeFor, basePosition, positionMatchup, teamDefense } from './defense-view.mjs';

const r = (team, position, stat, allowed, rank, of = 3) => ({ team, position, stat, allowed, rank, of });
const rows = [r('A', 'G', 'PTS', 20, 1), r('B', 'G', 'PTS', 15, 2), r('C', 'G', 'PTS', 10, 3), r('A', 'C', 'REB', 5, 3), r('A', 'C', 'PTS', 9, 2)];

test('grades run A (softest) to F', () => {
  assert.equal(gradeFor(1, 15), 'A');
  assert.equal(gradeFor(15, 15), 'F');
  assert.equal(gradeFor(1, 1), 'C');
});
test('basePosition keeps the first of G/F/C', () => {
  assert.equal(basePosition('g-f'), 'G');
  assert.equal(basePosition(''), null);
  assert.equal(basePosition(undefined), null);
});
test('positionMatchup returns rank, allowed and league mean', () => {
  const m = positionMatchup(rows, 'A', 'G');
  assert.equal(m.PTS.rank, 1);
  assert.equal(m.PTS.leagueAvg, 15);
  assert.equal(m.PTS.grade, 'A');
  assert.equal(m.REB, undefined);
});
test('teamDefense averages ranks and finds extremes', () => {
  const d = teamDefense(rows, 'A');
  assert.equal(d.byStat.PTS.rank, 2);
  assert.equal(d.softest.position, 'G');
  assert.equal(d.toughest.stat, 'REB');
  assert.equal(teamDefense(rows, 'Z'), null);
});
