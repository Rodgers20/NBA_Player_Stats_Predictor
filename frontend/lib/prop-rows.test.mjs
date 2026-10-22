import test from 'node:test';
import assert from 'node:assert/strict';
import { searchProps, rowCells, rowDashes, windowCell, sortByWindow } from './prop-rows.mjs';

test('dashes grade the last 10 results oldest first against the line and side', () => {
  const over = rowDashes({ line: 4.5, direction: 'Over', l10_values: [6, 4.5, 3, 5] }); // newest first
  assert.deepEqual(over.map(d => d.result), ['hit', 'miss', 'push', 'hit']);
  assert.deepEqual(over.map(d => d.value), [5, 3, 4.5, 6]);
  const under = rowDashes({ line: 4.5, direction: 'Under', l10_values: [6, 3] });
  assert.deepEqual(under.map(d => d.result), ['hit', 'miss']);
});

test('dashes never exceed ten and skip missing results', () => {
  const values = [null, ...Array.from({ length: 12 }, (_, i) => i)];
  assert.equal(rowDashes({ line: 5, direction: 'Over', l10_values: values }).length, 10);
});

test('no line or no results means no dashes, never a row of misses', () => {
  assert.deepEqual(rowDashes({ line: null, direction: 'Over', l10_values: [1, 2] }), []);
  assert.deepEqual(rowDashes({ line: 3, direction: 'Over' }), []);
  assert.deepEqual(rowDashes({ line: 3, direction: 'Over', l10_values: [] }), []);
});

const windowed = (l10, total = 10, h2h) => ({ hit_windows: { L5: { hits: 4, total: 5, pct: 80 }, L10: { hits: l10, total, pct: l10 * 10 }, L20: { hits: 12, total: 20, pct: 60 }, ...(h2h && { h2h }), season: { hits: 0, total: 0, pct: 0 } } });

test('a window cell is the chosen window, or null when it has no games', () => {
  assert.equal(windowCell(windowed(9), 'L10').pct, 90);
  assert.equal(windowCell(windowed(9), 'h2h'), null);
  assert.equal(windowCell(windowed(9), 'season'), null);
  assert.equal(windowCell({}, 'L10'), null);
});

test('sorting by a window ranks the highest rate first and keeps unknown windows last', () => {
  const rows = [
    { player: 'low', ...windowed(5) }, { player: 'none' }, { player: 'high', ...windowed(9) },
    { player: 'tie-small', ...windowed(9, 5) },
  ];
  assert.deepEqual(sortByWindow(rows, 'L10').map(r => r.player), ['high', 'tie-small', 'low', 'none']);
  assert.deepEqual(rows.map(r => r.player), ['low', 'none', 'high', 'tie-small']); // input untouched
});

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

test('a fallback without a hit count is not drawn at all, never as undefined', () => {
  assert.deepEqual(rowCells({ total: 10, hit_rate: 60 }), []);
});

test('without windows it falls back to one overall cell, and to nothing when there is no sample', () => {
  assert.deepEqual(rowCells({ hits: 7, total: 10, hit_rate: 70 }).map(c => [c.label, c.hits, c.total, c.pct, c.tone]), [['All', 7, 10, 70, 'good']]);
  assert.deepEqual(rowCells({ hits: 0, total: 0, hit_rate: null }), []);
  assert.deepEqual(rowCells({}), []);
});
