import test from 'node:test';
import assert from 'node:assert/strict';
import { localJournalAdd, localJournalList, localJournalSettle, oddsRange, summarizeBets } from './local-journal.mjs';

// Same fixture and expected numbers as tests/test_personal_bets_breakdown.py (parity).
const ROWS = [['PTS','Over',-110,'win','2026-10-01','A'], ['PTS','Under',120,'loss','2026-10-02','A'],
  ['REB','Over',-200,'loss','2026-10-03','B'], ['REB','Over',250,'push','2026-10-04','B'], ['AST','Over',150,'win','2026-10-05','A']];
const storage = () => { const map = new Map(); return { getItem:key => map.get(key) || null, setItem:(key,value) => map.set(key,value) }; };
function seeded() {
  const store = storage();
  ROWS.forEach(([stat, side, price, result, day, book], i) => {
    const { id } = localJournalAdd(store, { token:`t${i}`, mode:'paper', league:'nba', player:'P', game_date:day,
      stat, side, line:10.5, price, book, stake:10, notes:'' });
    localJournalSettle(store, id, result);
  });
  return localJournalList(store, 'paper').summary;
}
test('breakdowns by stat, side, book and odds range match the Python summary', () => {
  const s = seeded();
  assert.deepEqual(s.by_stat.map(r => r.key), ['PTS','REB','AST']);
  const pts = s.by_stat[0];
  assert.deepEqual([pts.bets, pts.wins, pts.losses, pts.pushes, pts.profit], [2,1,1,0,-0.91]);
  assert.equal(pts.roi.toFixed(2), '-4.55');
  assert.deepEqual(s.by_side.map(r => [r.key, r.bets, r.profit]), [['Over',4,14.09],['Under',1,-10]]);
  assert.deepEqual(s.by_book.map(r => [r.key, r.bets]), [['A',3],['B',2]]);
  assert.deepEqual(s.by_odds.map(r => [r.key, r.bets]), [['< -150',1],['-150 to +100',1],['+100 to +200',2],['> +200',1]]);
});
test('cumulative P&L curve and max drawdown match the Python summary', () => {
  const s = seeded();
  assert.deepEqual(s.curve.map(p => p.pnl), [9.09,-0.91,-10.91,-10.91,4.09]);
  assert.equal(s.max_drawdown, 20);
});
test('odds range boundaries and empty summary', () => {
  assert.deepEqual([-151,-150,100,101,200,201].map(oddsRange), ['< -150','-150 to +100','-150 to +100','+100 to +200','+100 to +200','> +200']);
  const empty = summarizeBets([]);
  assert.deepEqual([empty.by_stat, empty.curve, empty.max_drawdown], [[], [], 0]);
});
