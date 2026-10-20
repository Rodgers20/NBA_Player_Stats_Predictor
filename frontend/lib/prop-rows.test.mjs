import test from 'node:test';
import assert from 'node:assert/strict';
import { searchProps, rowCells } from './prop-rows.mjs';

const props = [
  { player: 'Cade Cunningham', team: 'DET', opponent: 'BOS' },
  { player: 'Jayson Tatum', team: 'BOS', opponent: 'DET' },
  { player: 'Jalen Duren', team: 'DET', opponent: null },
];

test('search matches player, team and opponent, ignoring case', () => {
  assert.deepEqual(searchProps(props, 'cade').map(p => p.player), ['Cade Cunningham']);
  assert.equal(searchProps(props, 'bos').length, 2);
  assert.equal(searchProps(props, 'DUR').length, 1);
});

test('an empty or blank search keeps everything', () => {
  assert.equal(searchProps(props, '').length, 3);
  assert.equal(searchProps(props, '   ').length, 3);
  assert.equal(searchProps(props, undefined).length, 3);
});

test('regex characters are plain text and never throw', () => {
  assert.deepEqual(searchProps(props, '.*'), []);
  assert.deepEqual(searchProps(props, '('), []);
  assert.deepEqual(searchProps(props, '[a-'), []);
});

test('cells come from the L5, L10 and L20 windows with hits out of games and a tone', () => {
  const cells = rowCells({ hit_windows: {
    L5: { hits: 4, total: 5, pct: 80 }, L10: { hits: 5, total: 10, pct: 50 }, L20: { hits: 5, total: 20, pct: 25 },
    season: { hits: 30, total: 50, pct: 60 }, h2h: { hits: 1, total: 2, pct: 50 },
  } });
  assert.deepEqual(cells.map(c => [c.label, c.hits, c.total, c.tone]), [['L5', 4, 5, 'good'], ['L10', 5, 10, 'neutral'], ['L20', 5, 20, 'bad']]);
});

test('a window with no games is skipped, never shown as zero percent', () => {
  const cells = rowCells({ hit_windows: { L5: { hits: 0, total: 0, pct: null }, L10: { hits: 6, total: 10, pct: 60 } } });
  assert.deepEqual(cells.map(c => c.label), ['L10']);
});

test('without windows it falls back to one overall cell, and to nothing when there is no sample', () => {
  assert.deepEqual(rowCells({ hits: 7, total: 10, hit_rate: 70 }).map(c => [c.label, c.hits, c.total, c.pct, c.tone]), [['All', 7, 10, 70, 'good']]);
  assert.deepEqual(rowCells({ hits: 0, total: 0, hit_rate: null }), []);
  assert.deepEqual(rowCells({}), []);
});
