import test from 'node:test';
import assert from 'node:assert/strict';
import { pivotDefense, sortRows, softness } from './defense-table.mjs';

const r = (team, position, stat, rank) => ({ team, position, stat, allowed: 20 - rank, rank, of: 3 });
const data = [r('AAA', 'G', 'PTS', 1), r('AAA', 'F', 'PTS', 3), r('BBB', 'G', 'PTS', 2), r('BBB', 'G', 'REB', 1), r('CCC', 'F', 'PTS', 1)];

test('pivot keeps one stat and averages ranks per team', () => {
  const { positions, rows } = pivotDefense(data, 'PTS');
  assert.deepEqual(positions, ['F', 'G']);
  assert.deepEqual(rows.map(x => [x.team, x.avgRank]), [['AAA', 2], ['BBB', 2], ['CCC', 1]]);
});

test('sorting is stable, immutable and puts missing cells last', () => {
  const { rows } = pivotDefense(data, 'PTS');
  assert.deepEqual(sortRows(rows, 'G').map(x => x.team), ['AAA', 'BBB', 'CCC']);
  assert.deepEqual(sortRows(rows, 'F', 'asc').map(x => x.team), ['CCC', 'AAA', 'BBB']);
  assert.deepEqual(sortRows(rows, 'team', 'desc').map(x => x.team), ['CCC', 'BBB', 'AAA']);
  assert.deepEqual(sortRows(rows, 'avg').map(x => x.team), ['CCC', 'AAA', 'BBB']);
  assert.equal(rows[0].team, 'AAA');
});

test('softness runs from softest to toughest', () => {
  assert.equal(softness(1, 30), 1); assert.equal(softness(30, 30), 0); assert.equal(softness(1, 1), 0.5);
});
