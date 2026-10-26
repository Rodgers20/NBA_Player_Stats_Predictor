import test from 'node:test';
import assert from 'node:assert/strict';
import { breakEven, playReason, quoteAge, tonightBoard, recordFor } from './tonight.mjs';

test('graded record is looked up by stat whatever order the combo parts are in', () => {
  const record = { by_stat: { PTS: { hit: 137, total: 247, pct: 55.5 }, 'PTS+AST+REB': { hit: 81, total: 140, pct: 57.9 }, BLK: { hit: 3, total: 12, pct: 25 } } };
  assert.deepEqual(recordFor(record, 'PTS'), { hit: 137, total: 247, pct: 55.5, early: false });
  assert.deepEqual(recordFor(record, 'PTS+REB+AST'), { hit: 81, total: 140, pct: 57.9, early: false });
  assert.equal(recordFor(record, 'BLK').early, true);
  assert.equal(recordFor(record, 'STL'), null);
  assert.equal(recordFor(undefined, 'PTS'), null);
  assert.equal(recordFor({ by_stat: {} }, 'PTS'), null);
});

test('break-even is the implied win rate of the price', () => {
  assert.equal(breakEven(-110).toFixed(1), '52.4');
  assert.equal(breakEven(-200).toFixed(1), '66.7');
  assert.equal(breakEven(150).toFixed(1), '40.0');
  assert.equal(breakEven(null), null);
  assert.equal(breakEven(0), null);
});

test('quote age is relative to now and absent without a timestamp', () => {
  const now = Date.parse('2026-10-19T18:00:00Z');
  assert.deepEqual(quoteAge('2026-10-19T17:52:00Z', now), { label: '8 min ago', stale: false });
  assert.deepEqual(quoteAge('2026-10-19T15:00:00Z', now), { label: '3 hr ago', stale: true });
  assert.deepEqual(quoteAge('2026-10-19T17:59:50Z', now), { label: 'just now', stale: false });
  assert.equal(quoteAge(null, now), null);
  assert.equal(quoteAge('not a date', now), null);
});

test('board ranks priced picks by EV and keeps leans and research apart', () => {
  const props = [
    { player: 'A', pick_type: 'lean', ev: -0.01 },
    { player: 'B', pick_type: 'pick', ev: 0.04 },
    { player: 'C', pick_type: 'research', ev: null },
    { player: 'D', pick_type: 'pick', ev: 0.09 },
    { player: 'E', ev: 0.2 },
  ];
  const board = tonightBoard(props);
  assert.deepEqual(board.picks.map(p => p.player), ['D', 'B']);
  assert.deepEqual(board.leans.map(p => p.player), ['A']);
  assert.deepEqual(board.research.map(p => p.player), ['C']);
});

test('a play explains itself in one sentence, and says so when recent form disagrees', () => {
  const base = { direction: 'Over', line: 4.5, model_projection: 5.2, stat_label: 'AST', model_prob: 0.52, price: 118 };
  assert.match(playReason({ ...base, hits: 3, total: 10 }), /^Against recent form: cleared 3 of 10, but the model projects 5\.2 on a 4\.5 line\.$/);
  assert.match(playReason({ ...base, hits: 8, total: 10 }), /^Form agrees: cleared 8 of 10, and the model projects 5\.2 on a 4\.5 line\.$/);
  assert.match(playReason({ ...base, hits: 5, total: 10 }), /^Model projects 5\.2 on a 4\.5 line: 52% for this side, the price needs 45\.9%\.$/);
  assert.equal(playReason({ ...base, model_projection: null, hits: 5, total: 10 }), null);
});
