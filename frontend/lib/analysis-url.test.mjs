import test from 'node:test';
import assert from 'node:assert/strict';
import { parseAnalysisView, analysisViewToSearch } from './analysis-url.mjs';

const defaults = { stat: 'PTS', period: 10, line: null };

test('an empty query gives the defaults and writes nothing', () => {
  assert.deepEqual(parseAnalysisView('', defaults), defaults);
  assert.equal(analysisViewToSearch(defaults, defaults), '');
});

test('a view survives the round trip, including combo stats and named windows', () => {
  const view = { stat: 'PTS+REB+AST', period: 'h2h', line: 24.5 };
  assert.deepEqual(parseAnalysisView(analysisViewToSearch(view, defaults), defaults), view);
  assert.deepEqual(parseAnalysisView('stat=PTS REB&period=20&line=7', defaults), { stat: 'PTS+REB', period: 20, line: 7 });
});

test('only non-default parts are written, in a stable order', () => {
  assert.equal(analysisViewToSearch({ stat: 'AST', period: 10, line: 5.5 }, defaults), 'stat=AST&line=5.5');
});

test('junk falls back to the defaults', () => {
  assert.deepEqual(parseAnalysisView('stat=ZZZ&period=7&line=abc', defaults), defaults);
  assert.equal(parseAnalysisView('line=-3', defaults).line, null);
  assert.equal(parseAnalysisView('line=999', defaults).line, null);
  assert.equal(parseAnalysisView('line=Infinity', defaults).line, null);
  assert.equal(parseAnalysisView('line=', defaults).line, null);
});

test('lines snap to half points', () => {
  assert.equal(parseAnalysisView('line=24.3', defaults).line, 24.5);
});

test('first-quarter stats survive a shared analysis link', () => {
  assert.equal(parseAnalysisView('stat=PTS_1Q&line=6.5', { stat: 'PTS', period: 10, line: null }).stat, 'PTS_1Q');
  assert.equal(parseAnalysisView('stat=STL_1Q', { stat: 'PTS', period: 10, line: null }).stat, 'PTS', 'only quarters we collect');
});
