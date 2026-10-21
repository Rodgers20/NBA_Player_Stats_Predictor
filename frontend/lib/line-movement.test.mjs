import test from 'node:test';
import assert from 'node:assert/strict';
import { bookSeries, primaryBook, movement } from './line-movement.mjs';

const snaps = [
  { book: 'DK', line: 5.5, ts: 1 }, { book: 'FD', line: 5.5, ts: 2 },
  { book: 'DK', line: 6, ts: 3 }, { book: 'DK', line: 6.5, ts: 4 },
];

test('primary book is the one with the most snapshots', () => {
  assert.equal(primaryBook(snaps), 'DK');
  assert.equal(primaryBook([]), null);
  assert.deepEqual([...bookSeries(snaps).keys()], ['DK', 'FD']);
});

test('movement reports opening, current and change', () => {
  assert.deepEqual(movement(bookSeries(snaps).get('DK')), { opening: 5.5, current: 6.5, change: 1, samples: 3 });
  assert.equal(movement([]), null);
});
