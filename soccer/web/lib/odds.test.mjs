import test from "node:test";
import assert from "node:assert/strict";
import { impliedProbs, modelVsBook } from "./odds.mjs";

test("implied probabilities remove the margin and sum to 1", () => {
  const p = impliedProbs({ home: 2.0, draw: 3.5, away: 4.0 });
  assert.ok(Math.abs(p.home + p.draw + p.away - 1) < 1e-9);
  assert.ok(p.home > p.draw && p.draw > p.away);
});

test("missing or invalid prices give null", () => {
  assert.equal(impliedProbs({ home: 2, draw: null, away: 3 }), null);
  assert.equal(impliedProbs({ home: 1, draw: 3, away: 3 }), null);
  assert.equal(impliedProbs(undefined), null);
});

test("model vs book flags the biggest gap of at least 5 points", () => {
  const odds = { home: 2.0, draw: 3.5, away: 4.0 };
  const book = impliedProbs(odds);
  const win = { home: book.home + 0.12, draw: book.draw - 0.07, away: book.away - 0.05 };
  const flag = modelVsBook(win, odds);
  assert.equal(flag.side, "home");
  assert.ok(flag.gap > 0.11);
});

test("no flag when the model agrees with the book", () => {
  const odds = { home: 2.0, draw: 3.5, away: 4.0 };
  assert.equal(modelVsBook(impliedProbs(odds), odds), null);
});
