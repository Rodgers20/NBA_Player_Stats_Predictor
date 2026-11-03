import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_FILTERS, parseFilters, toQuery, applyFilters, marketCounts, takeRows, totalRows, teamsOf, probTone,
} from "./props-filters.mjs";

const p = (o) => ({ player: "A", team: "Arsenal", is_home: true, market: "anytime_goal", line: null, prob: 0.4,
  expected: 0.5, per_game: 0.4, confidence: "high", ...o });
const fixtures = [
  { fixture_key: "f1", home_team: "Arsenal", away_team: "Chelsea", props: [
    p({ player: "Saka", prob: 0.41 }),
    p({ player: "Havertz", prob: 0.33, confidence: "medium" }),
    p({ player: "Palmer", team: "Chelsea", is_home: false, prob: 0.38, confidence: "low" }),
    p({ player: "Saka", market: "shots_on_target", line: 0.5, prob: 0.72 }),
    p({ player: "Saka", market: "shots_on_target", line: 1.5, prob: 0.36 }),
    p({ player: "Odegaard", market: "assist", prob: 0.18, confidence: "low" }),
  ] },
  { fixture_key: "f2", home_team: "Leeds", away_team: "Spurs", props: [
    p({ player: "Kane", team: "Spurs", is_home: false, prob: 0.55 }),
  ] },
];

test("parse defaults and bad values", () => {
  assert.deepEqual(parseFilters(new URLSearchParams("")), DEFAULT_FILTERS);
  const f = parseFilters(new URLSearchParams("m=nope&line=9&v=x&c=x&p=500&s=x"));
  assert.equal(f.market, "all");
  assert.equal(f.line, null);
  assert.equal(f.venue, "all");
  assert.equal(f.conf, "all");
  assert.equal(f.minProb, 90);
  assert.equal(f.sort, "prob");
});

test("parse accepts Next searchParams objects with arrays", () => {
  const f = parseFilters({ m: "sot", line: "1.5", t: ["Arsenal", "Spurs"], f: ["f2", "f1"] });
  assert.equal(f.market, "shots_on_target");
  assert.equal(f.line, 1.5);
  assert.deepEqual(f.teams, ["Arsenal", "Spurs"]);
  assert.equal(f.fixture, "f2");
});

test("toQuery round-trips and omits defaults", () => {
  assert.equal(toQuery(DEFAULT_FILTERS), "");
  const f = { ...DEFAULT_FILTERS, market: "shots_on_target", line: 0.5, venue: "home", conf: "medium",
    minProb: 30, teams: ["Arsenal", "A & B"], fixture: "f1", sort: "expected" };
  assert.deepEqual(parseFilters(new URLSearchParams(toQuery(f))), f);
});

test("market filter and sorting by probability", () => {
  const out = applyFilters(fixtures, { ...DEFAULT_FILTERS, market: "anytime_goal" });
  assert.deepEqual(out[0].props.map((x) => x.player), ["Saka", "Palmer", "Havertz"]);
  assert.equal(out.length, 2);
});

test("line only constrains shots on target", () => {
  const out = applyFilters(fixtures, { ...DEFAULT_FILTERS, line: 1.5 });
  const rows = out[0].props;
  assert.ok(rows.some((x) => x.market === "anytime_goal"));
  assert.deepEqual(rows.filter((x) => x.market === "shots_on_target").map((x) => x.line), [1.5]);
});

test("venue, confidence, min probability and team filters", () => {
  assert.ok(applyFilters(fixtures, { ...DEFAULT_FILTERS, venue: "away" }).every((fx) => fx.props.every((x) => !x.is_home)));
  const med = applyFilters(fixtures, { ...DEFAULT_FILTERS, conf: "medium" });
  assert.ok(med.every((fx) => fx.props.every((x) => x.confidence !== "low")));
  const hi = applyFilters(fixtures, { ...DEFAULT_FILTERS, minProb: 50 });
  assert.deepEqual(hi.flatMap((fx) => fx.props.map((x) => x.player)), ["Saka", "Kane"]);
  const team = applyFilters(fixtures, { ...DEFAULT_FILTERS, teams: ["Spurs"] });
  assert.deepEqual(team.map((fx) => fx.fixture_key), ["f2"]);
});

test("fixture filter and non-mutation", () => {
  const before = JSON.stringify(fixtures);
  const out = applyFilters(fixtures, { ...DEFAULT_FILTERS, fixture: "f2" });
  assert.deepEqual(out.map((fx) => fx.fixture_key), ["f2"]);
  assert.equal(JSON.stringify(fixtures), before);
});

test("sort by expected and per-game", () => {
  const fx = [{ fixture_key: "x", home_team: "a", away_team: "b", props: [
    p({ player: "L", prob: 0.9, expected: 0.1, per_game: 0.9 }),
    p({ player: "H", prob: 0.1, expected: 0.9, per_game: 0.1 }),
  ] }];
  assert.equal(applyFilters(fx, { ...DEFAULT_FILTERS, sort: "expected" })[0].props[0].player, "H");
  assert.equal(applyFilters(fx, { ...DEFAULT_FILTERS, sort: "avg" })[0].props[0].player, "L");
});

test("marketCounts ignore the market filter but respect others", () => {
  const c = marketCounts(fixtures, { ...DEFAULT_FILTERS, market: "assist", venue: "home" });
  assert.deepEqual(c, { all: 5, anytime_goal: 2, shots_on_target: 2, assist: 1 });
});

test("takeRows truncates across fixtures", () => {
  const out = takeRows(fixtures, 7);
  assert.equal(totalRows(out), 7);
  assert.equal(out.length, 2);
  assert.equal(totalRows(takeRows(fixtures, 3)), 3);
  assert.equal(takeRows(fixtures, 0).length, 0);
});

test("teamsOf and probTone", () => {
  assert.deepEqual(teamsOf(fixtures), ["Arsenal", "Chelsea", "Leeds", "Spurs"]);
  assert.equal(probTone(0.5), "hot");
  assert.equal(probTone(0.3), "warm");
  assert.equal(probTone(0.29), "cool");
});
