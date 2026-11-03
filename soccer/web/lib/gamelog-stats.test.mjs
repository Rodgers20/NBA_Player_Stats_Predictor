import test from "node:test";
import assert from "node:assert/strict";
import {
  statValue, filterGames, valuesOf, hitRate, average, median, hitHeadline,
  fairOdds, modelProbFor, nearestLine, fmtNum,
} from "./gamelog-stats.mjs";

const games = [
  { opponent: "Wolves", is_home: true, goals: 1, assists: 0, shots: 4, sot: 2, fouls_committed: 1, yellow: 0, red: 0 },
  { opponent: "Chelsea", is_home: false, goals: 0, assists: 1, shots: 2, sot: 0, fouls_committed: 3, yellow: 1, red: 0 },
  { opponent: "Hull", is_home: true, goals: 2, assists: 0, shots: 5, sot: 3, fouls_committed: 0, yellow: 0, red: 0 },
  { opponent: "Chelsea", is_home: true, goals: 0, assists: 0, shots: 1, sot: 0, fouls_committed: 2, yellow: 0, red: 1 },
  { opponent: "Spurs", is_home: false, goals: 1, assists: 0, shots: 3, sot: 1, fouls_committed: 1, yellow: 0, red: 0 },
  { opponent: "Leeds", is_home: false, goals: 0, assists: 0, shots: 0, sot: 0 },
];

test("statValue maps stats, sums cards, returns null for missing fields", () => {
  assert.equal(statValue(games[0], "sot"), 2);
  assert.equal(statValue(games[1], "cards"), 1);
  assert.equal(statValue(games[3], "cards"), 1);
  assert.equal(statValue(games[5], "fouls"), null);
  assert.equal(statValue(games[5], "cards"), null);
});

test("filterGames windows and venues", () => {
  assert.equal(filterGames(games, "L5").length, 5);
  assert.equal(filterGames(games, "L10").length, 6);
  assert.equal(filterGames(games, "Home").length, 3);
  assert.equal(filterGames(games, "Away").length, 3);
  assert.deepEqual(filterGames(games, "VS", "chelsea").map((g) => g.is_home), [false, true]);
  assert.equal(filterGames(games, "VS", null).length, 0);
  assert.equal(filterGames(games, "Season").length, 6);
});

test("valuesOf drops games missing the stat", () => {
  assert.deepEqual(valuesOf(games, "fouls"), [1, 3, 0, 2, 1]);
});

test("hitRate over/under and pushes", () => {
  const v = valuesOf(games, "goals"); // 1,0,2,0,1,0
  assert.deepEqual(hitRate(v, 0.5), { hits: 3, pushes: 0, total: 6, rate: 0.5 });
  assert.deepEqual(hitRate(v, 0.5, "under"), { hits: 3, pushes: 0, total: 6, rate: 0.5 });
  assert.deepEqual(hitRate(v, 1), { hits: 1, pushes: 2, total: 6, rate: 1 / 6 });
  assert.deepEqual(hitRate([], 0.5), { hits: 0, pushes: 0, total: 0, rate: null });
});

test("average, median, fmtNum", () => {
  assert.equal(average([1, 2, 3, 4]), 2.5);
  assert.equal(average([]), null);
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(median([]), null);
  assert.equal(fmtNum(2), "2");
  assert.equal(fmtNum(1.234, 2), "1.23");
  assert.equal(fmtNum(null), "—");
});

test("hitHeadline", () => {
  assert.equal(hitHeadline({ hits: 5, pushes: 0, total: 8, rate: 0.625 }, 0.5), "63% — 5 of 8 games over 0.5");
  assert.equal(hitHeadline({ hits: 0, pushes: 0, total: 0, rate: null }, 0.5), "No games in this window");
});

test("fairOdds", () => {
  assert.deepEqual(fairOdds(0.5), { decimal: 2, american: "-100" });
  assert.deepEqual(fairOdds(0.25), { decimal: 4, american: "+300" });
  assert.deepEqual(fairOdds(0.8), { decimal: 1.25, american: "-400" });
  assert.equal(fairOdds(0), null);
  assert.equal(fairOdds(1), null);
});

test("modelProbFor matches market and line", () => {
  const props = [
    { market: "anytime_goal", line: null, prob: 0.41 },
    { market: "shots_on_target", line: 0.5, prob: 0.7 },
    { market: "shots_on_target", line: 1.5, prob: 0.33 },
  ];
  assert.equal(modelProbFor(props, "goals", 0.5).prob, 0.41);
  assert.equal(modelProbFor(props, "goals", 1.5), null);
  assert.equal(modelProbFor(props, "sot", 1.5).prob, 0.33);
  assert.equal(modelProbFor(props, "sot", 1.5, "under").prob, 0.67);
  assert.equal(modelProbFor(props, "shots", 1.5), null);
  assert.equal(modelProbFor(null, "goals", 0.5), null);
});

test("nearestLine snaps to an allowed line", () => {
  assert.equal(nearestLine("sot", 1.4), 1.5);
  assert.equal(nearestLine("cards", 3), 0.5);
});
