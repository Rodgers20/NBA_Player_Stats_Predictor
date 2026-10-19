import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPriceRequest, describeCorrelation, relatedProps } from './slip-price.mjs';

const leg = (player, stat, extra = {}) => ({ league: 'nba', date: '2026-10-20', player, stat, direction: 'Over', line: 20.5,
  price: -110, probability: 0.56, matchup: 'LAL vs BOS', ...extra });

test('request maps legs, resolves teams from loaded props and builds a shared game id', () => {
  const body = buildPriceRequest([leg('A B', 'PTS'), leg('C D', 'AST', { direction: 'Under' })], { 'a b': 'LAL' });
  assert.deepEqual(body.legs[0], { player: 'A B', team: 'LAL', game: 'nba:2026-10-20:LAL vs BOS', stat: 'PTS', side: 'Over', prob: 0.56, price: -110 });
  assert.equal(body.legs[1].team, '');
  assert.equal(body.legs[1].side, 'Under');
});
test('slips that cannot be priced return null so the page keeps the independent calculation', () => {
  assert.equal(buildPriceRequest([leg('A', 'PTS')]), null);
  assert.equal(buildPriceRequest([leg('A', 'PTS'), leg('B', 'PTS', { probability: null })]), null);
  assert.equal(buildPriceRequest(Array.from({ length: 13 }, (_, i) => leg(`P${i}`, 'PTS'))), null);
  assert.equal(buildPriceRequest([leg('A', 'PTS'), leg('B', 'PTS', { matchup: '' })]).legs[1].game, '');
});
test('correlation note text', () => {
  assert.equal(describeCorrelation({ notes: [] }), null);
  assert.equal(describeCorrelation({ notes: [{ with: 2, rho: 0.6, reason: 'same player' }, { with: 3, rho: -0.05, reason: 'teammates' }] }),
    'Correlated with leg 2 (+0.60): same player. Correlated with leg 3 (-0.05): teammates');
});
test('related props are the same player sharing a stat, not already on the slip', () => {
  const props = [
    { player: 'A B', stat: 'PTS+REB+AST', direction: 'Over', line: 35.5, price: -115 },
    { player: 'A B', stat: 'REB', direction: 'Over', line: 6.5, price: -110 },
    { player: 'A B', stat: 'PTS', direction: 'Over', line: 20.5, price: -110 },
    { player: 'Z Z', stat: 'PTS+REB', direction: 'Over', line: 30.5, price: -110 },
    { player: 'A B', stat: 'PTS+AST', direction: 'Over', line: null, price: -110 },
  ];
  const slip = [leg('A B', 'PTS'), leg('A B', 'REB')];
  assert.deepEqual(relatedProps(slip[0], props, slip).map(p => p.stat), ['PTS+REB+AST']);
  assert.deepEqual(relatedProps(slip[0], props, [slip[0]]).map(p => p.stat), ['PTS+REB+AST']);
});
