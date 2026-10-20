import test from 'node:test';
import assert from 'node:assert/strict';
import { safeTeamColor, teamGlow } from './team-glow.mjs';

test('a valid six digit hex is normalised to lowercase with a hash', () => {
  assert.equal(safeTeamColor('1D428A'), '#1d428a');
  assert.equal(safeTeamColor('#552583'), '#552583');
});

test('missing or malformed colours fall back and never pass raw text through', () => {
  for (const bad of ['#GGGGGG', '', null, undefined, 'red; background:url(x)', '#fff'])
    assert.equal(safeTeamColor(bad), '#26434d');
});

test('the glow gradient is built only from the safe colour', () => {
  const glow = teamGlow('bad');
  assert.ok(glow.includes('#26434d'));
  assert.ok(!glow.includes('url('));
  assert.ok(teamGlow('1d428a').includes('#1d428a'));
});
