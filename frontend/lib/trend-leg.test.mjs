import test from 'node:test';
import assert from 'node:assert/strict';
import { trendLeg } from './trend-leg.mjs';
import { legKey, toggleLeg } from './bet-slip.mjs';

const card = (extra = {}) => ({ player: 'Cade Cunningham', team: 'DET', opponent: 'BOS', stat: 'PTS', side: 'over', line: 22.5, hits: 8, games: 10,
  results: [], avg: 25, text: '', stars: 4, line_source: 'derived', live: null, ...extra });
const ctx = { league: 'nba', date: '2026-10-20' };
const liveCard = (price = -115) => card({ line_source: 'live', live: { line: 24.5, price, book: 'FanDuel' } });

test('a trend with a live sportsbook line uses that line and its real price', () => {
  const leg = trendLeg({ card: liveCard(), ...ctx });
  assert.deepEqual({ player: leg.player, stat: leg.stat, direction: leg.direction, line: leg.line, price: leg.price, probability: leg.probability, matchup: leg.matchup, team: leg.team },
    { player: 'Cade Cunningham', stat: 'PTS', direction: 'Over', line: 24.5, price: -115, probability: null, matchup: 'DET vs BOS', team: 'DET' });
});

test('a derived trend has no sportsbook price, so it needs the one the user enters and is never given a guess', () => {
  assert.equal(trendLeg({ card: card(), ...ctx }), null);
  const leg = trendLeg({ card: card(), ...ctx, price: -110 });
  assert.equal(leg.line, 22.5);
  assert.equal(leg.price, -110);
});

test('a price the user enters wins over the live price, and a live card keeps its live line otherwise', () => {
  assert.equal(trendLeg({ card: liveCard(), ...ctx, price: 120 }).price, 120);
  assert.equal(trendLeg({ card: liveCard(), ...ctx }).price, -115);
});

test('a live line with no price still needs the user price, on the live line', () => {
  assert.equal(trendLeg({ card: liveCard(null), ...ctx }), null);
  assert.equal(trendLeg({ card: liveCard(null), ...ctx, price: -105 }).line, 24.5);
});

test('junk prices and lines are refused', () => {
  for (const bad of [0, 50, -50, NaN, Infinity, '-110x']) assert.equal(trendLeg({ card: card(), ...ctx, price: bad }), null);
  assert.equal(trendLeg({ card: card({ line: NaN }), ...ctx, price: -110 }), null);
});

test('unders stay unders, and a missing opponent falls back to the team', () => {
  const leg = trendLeg({ card: card({ side: 'under', opponent: null }), ...ctx, price: -110 });
  assert.equal(leg.direction, 'Under');
  assert.equal(leg.matchup, 'DET');
});

test('the same player and stat toggles between over and under on the slip', () => {
  const over = trendLeg({ card: card(), ...ctx, price: -110 });
  const under = trendLeg({ card: card({ side: 'under' }), ...ctx, price: -110 });
  assert.equal(legKey(over), legKey(under));
  assert.deepEqual(toggleLeg([over], under).map(leg => leg.direction), ['Under']);
});
