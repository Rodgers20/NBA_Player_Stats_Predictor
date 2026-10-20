import test from 'node:test';
import assert from 'node:assert/strict';
import { parseFilters, filtersToSearch } from './props-url.mjs';

const defaults = { stat: '', direction: 'All', location: 'All', sort: 'hit_rate', game: '', locksOnly: false, minStars: 0 };

test('an empty query gives the defaults and an empty search', () => {
  assert.deepEqual(parseFilters('', defaults), { ...defaults, q: '' });
  assert.equal(filtersToSearch({ ...defaults, q: '' }, defaults), '');
});

test('filters survive a round trip through the URL, including combo stats with a plus sign', () => {
  const filters = { ...defaults, stat: 'PTS+AST', direction: 'Under', location: 'Home', sort: 'edge', game: 'BOS @ DET', minStars: 3, q: 'cade' };
  const search = filtersToSearch(filters, defaults);
  assert.deepEqual(parseFilters(search, defaults), filters);
  assert.ok(!search.includes(' ') && !search.includes('+AST'), 'plus and spaces are encoded');
});

test('only non-default values are written, in a stable order', () => {
  assert.equal(filtersToSearch({ ...defaults, sort: 'ev', stat: 'PTS', q: '' }, defaults), 'stat=PTS&sort=ev');
});

test('a plus turned into a space by a hand-typed URL still means the combo stat', () => {
  assert.equal(parseFilters('stat=PTS AST', defaults).stat, 'PTS+AST');
});

test('junk values fall back to the defaults and never pass through', () => {
  const parsed = parseFilters('stat=ZZZ&direction=sideways&location=moon&sort=drop%20table&minStars=99', defaults);
  assert.deepEqual(parsed, { ...defaults, q: '' });
  assert.equal(parseFilters('minStars=-4', defaults).minStars, 0);
  assert.equal(parseFilters('minStars=2.7', defaults).minStars, 0);
  assert.equal(parseFilters('minStars=abc', defaults).minStars, 0);
});

test('search text is trimmed, stripped of control characters and capped', () => {
  assert.equal(parseFilters('q=%20%20lebron%0A%20', defaults).q, 'lebron');
  assert.equal(parseFilters(`q=${'x'.repeat(100)}`, defaults).q.length, 40);
  assert.equal(parseFilters(`game=${'g'.repeat(200)}`, defaults).game.length, 60);
});

test('a leading question mark is accepted', () => {
  assert.equal(parseFilters('?stat=REB', defaults).stat, 'REB');
});
