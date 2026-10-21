import test from "node:test";
import assert from "node:assert/strict";
import { formatOdds, edgeTier, hitTone, hitChips, bestBookKeys, filterMinStars, sortByEdge } from "./prop-research.mjs";

test("formatOdds", () => {
  assert.equal(formatOdds(110), "+110");
  assert.equal(formatOdds(-115.4), "-115");
  assert.equal(formatOdds(null), null);
});
test("edgeTier ladder", () => {
  assert.deepEqual([edgeTier(null), edgeTier(-1), edgeTier(0.5), edgeTier(1), edgeTier(2.9), edgeTier(3)], [null, 0, 0, 1, 2, 3]);
});
test("hitTone thresholds", () => {
  assert.deepEqual([hitTone(70), hitTone(69), hitTone(41), hitTone(40), hitTone(null)], ["good", "neutral", "neutral", "bad", "neutral"]);
});
test("hitChips skips empty windows and tolerates null", () => {
  assert.deepEqual(hitChips(null), []);
  const chips = hitChips({ L5: { hits: 4, total: 5, pct: 80 }, L10: { hits: 0, total: 0, pct: null }, L20: { hits: 5, total: 20, pct: 25 }, season: { hits: 6, total: 10, pct: null }, h2h: { hits: 0, total: 0, pct: null } });
  assert.deepEqual(chips.map((c) => [c.label, c.pct, c.tone]), [["L5", 80, "good"], ["L20", 25, "bad"], ["Season", 60, "neutral"]]);
});
test("bestBookKeys handles ties and missing prices", () => {
  const books = [{ book: "a", line: 1, over_price: -110, under_price: null }, { book: "b", line: 1, over_price: 100, under_price: -120 }, { book: "c", line: 1, over_price: 100, under_price: null }];
  assert.deepEqual([...bestBookKeys(books, "over")], ["b", "c"]);
  assert.deepEqual([...bestBookKeys(books, "under")], ["b"]);
  assert.equal(bestBookKeys(undefined, "over").size, 0);
});
test("filterMinStars / sortByEdge", () => {
  const ps = [{ stars: 2, edge_pct: 1 }, { stars: null }, { stars: 4, edge_pct: 5 }];
  assert.equal(filterMinStars(ps, 0).length, 3);
  assert.equal(filterMinStars(ps, 3).length, 1);
  assert.deepEqual(sortByEdge(ps).map((p) => p.edge_pct), [5, 1, undefined]);
});
