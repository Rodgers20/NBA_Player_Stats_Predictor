import test from 'node:test';
import assert from 'node:assert/strict';
import { localJournalAdd, localJournalList, localJournalSettle } from './local-journal.mjs';

const storage = () => { const map = new Map(); return { getItem:key => map.get(key) || null, setItem:(key,value) => map.set(key,value) }; };
test('browser journal saves idempotently and settles without a server', () => {
  const store = storage();
  const entry = { token:'same', mode:'paper', league:'wnba', player:'A Player', game_date:'2026-10-03',
    stat:'PTS', side:'Over', line:19.5, price:-110, book:'Paper', stake:11, notes:'' };
  const first = localJournalAdd(store, entry);
  assert.deepEqual(localJournalAdd(store, entry), first);
  assert.equal(localJournalList(store, 'paper').bets.length, 1);
  localJournalSettle(store, first.id, 'win');
  const journal = localJournalList(store, 'paper');
  assert.equal(journal.bets[0].profit_cents, 1000);
  assert.equal(journal.summary.roi, 1000 / 1100 * 100);
  assert.equal(localJournalList(store, 'real').bets.length, 0);
});
