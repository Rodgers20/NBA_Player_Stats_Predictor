import test from 'node:test';
import assert from 'node:assert/strict';
import { spreadText, disagrees, spreadEdge, defaultDay, dayRecord, filterGames, tipText, barColor, mark } from './cbb.mjs';

const team = (over = {}) => ({ rank: null, conference: 'SEC', color: '0021a5', alt_color: 'fa4616', ...over });
const game = (over = {}) => ({ home: team(), away: team(), line: { spread_home: -4.5 }, pick: { winner: 'home', market_p_home: 0.64, proj_margin: 6 }, result: null, ...over });

test('spreadText flips sign for the away side and shows PK', () => {
  assert.equal(spreadText(-4.5, 'home'), '-4.5');
  assert.equal(spreadText(-4.5, 'away'), '+4.5');
  assert.equal(spreadText(0, 'away'), 'PK');
  assert.equal(spreadText(null, 'home'), null);
});

test('disagrees only when model and market pick different winners', () => {
  assert.equal(disagrees(game()), false);
  assert.equal(disagrees(game({ pick: { winner: 'away', market_p_home: 0.64 } })), true);
  assert.equal(disagrees(game({ pick: { winner: 'away', market_p_home: null } })), false);
});

test('spreadEdge is model margin vs the book, home minus away', () => {
  assert.equal(spreadEdge(game()), 1.5);
  assert.equal(spreadEdge(game({ line: null })), null);
});

test('defaultDay prefers today, else nearest day with games, ties go later', () => {
  const days = [{ date: '2026-10-30', games: 0 }, { date: '2026-10-31', games: 3 }, { date: '2026-11-02', games: 5 }];
  assert.equal(defaultDay(days, '2026-11-02'), '2026-11-02');
  assert.equal(defaultDay(days, '2026-11-01'), '2026-11-02');
  assert.equal(defaultDay(days, '2026-10-30'), '2026-10-31');
  assert.equal(defaultDay([], '2026-11-01'), '2026-11-01');
});

test('dayRecord counts hits and misses and skips pushes', () => {
  const r = (winner_correct, ats_correct, total_correct) => game({ result: { winner_correct, ats_correct, total_correct } });
  const rec = dayRecord([r(true, false, null), r(false, true, true), game()]);
  assert.deepEqual(rec, { winner: { wins: 1, losses: 1 }, ats: { wins: 1, losses: 1 }, total: { wins: 1, losses: 0 }, graded: 2 });
});

test('filterGames handles top25, lines and conference', () => {
  const ranked = game({ home: team({ rank: 3 }) });
  const other = game({ line: null, away: team({ conference: 'ACC' }), home: team({ conference: 'ACC' }) });
  assert.deepEqual(filterGames([ranked, other], { view: 'top25' }), [ranked]);
  assert.deepEqual(filterGames([ranked, other], { view: 'lines' }), [ranked]);
  assert.deepEqual(filterGames([ranked, other], { conference: 'ACC' }), [other]);
});

test('tipText formats 24h Eastern times', () => {
  assert.equal(tipText('18:00'), '6:00 PM ET');
  assert.equal(tipText('00:05'), '12:05 AM ET');
  assert.equal(tipText(null), 'TBA');
});

test('barColor falls back from a dark primary to the alternate colour', () => {
  assert.equal(barColor({ color: '0021a5', alt_color: 'fa4616' }), '#fa4616');
  assert.equal(barColor({ color: 'c8102e', alt_color: '000000' }), '#c8102e');
  assert.equal(barColor({ color: '000000', alt_color: '000000' }), '#7aa7b0');
});

test('mark maps correctness to hit/miss/push', () => {
  assert.equal(mark(true), 'hit');
  assert.equal(mark(false), 'miss');
  assert.equal(mark(null, 'push'), 'push');
  assert.equal(mark(null, null), null);
});
