import test from 'node:test';
import assert from 'node:assert/strict';
import { handoff, handoffLeg, safeBookUrl } from './sportsbook-links.mjs';

const FD_GAME = 'https://sportsbook.fanduel.com/basketball/nba/heat-@-celtics-1';
const fdSlip = (market, selection) => `https://sportsbook.fanduel.com/addToBetslip?marketId=${market}&selectionId=${selection}`;
const leg = (player, stat, line, price, direction = 'Over') => ({ league: 'nba', date: '2026-10-27', player, stat, direction, line, price, probability: null, matchup: 'MIA @ BOS', addedAt: '' });
const board = { props: [
  { player: 'Jayson Tatum', stat: 'PTS', books: [
    { book: 'FanDuel', line: 27.5, over_price: -115, under_price: -105, event_link: FD_GAME, over_link: fdSlip('42.1', '7'), under_link: fdSlip('42.1', '8') },
    { book: 'DraftKings', line: 28.5, over_price: -110, under_price: -110, event_link: 'https://sportsbook.draftkings.com/event/9' }] },
  { player: 'Jaylen Brown', stat: 'REB', books: [
    { book: 'FanDuel', line: 6.5, over_price: 120, under_price: -150, event_link: FD_GAME, over_link: fdSlip('42.2', '11') }] },
  { player: 'Bam Adebayo', stat: 'AST', books: [] },
] };

test('only https links on the book’s own domain are opened', () => {
  assert.equal(safeBookUrl(fdSlip('1', '2'), 'fanduel'), fdSlip('1', '2'));
  assert.equal(safeBookUrl('https://nj.sportsbook.fanduel.com/x', 'fanduel'), 'https://nj.sportsbook.fanduel.com/x');
  assert.equal(safeBookUrl('http://sportsbook.fanduel.com/x', 'fanduel'), null);
  assert.equal(safeBookUrl('https://fanduel.com.evil.io/x', 'fanduel'), null);
  assert.equal(safeBookUrl('javascript:alert(1)', 'fanduel'), null);
});

test('a leg at the book’s line opens straight into the slip', () => {
  const result = handoffLeg(leg('Jayson Tatum', 'PTS', 27.5, -115), board, 'fanduel');
  assert.deepEqual([result.status, result.depth, result.url], ['ready', 'slip', fdSlip('42.1', '7')]);
  assert.equal(handoffLeg(leg('Jayson Tatum', 'PTS', 27.5, -105, 'Under'), board, 'fanduel').url, fdSlip('42.1', '8'));
});

test('a changed price is flagged but still opens the same selection', () => {
  const result = handoffLeg(leg('Jayson Tatum', 'PTS', 27.5, -130), board, 'fanduel');
  assert.deepEqual([result.status, result.depth, result.price], ['price', 'slip', -115]);
});

test('a moved line never opens a different selection: it falls back to the game page', () => {
  const result = handoffLeg(leg('Jayson Tatum', 'PTS', 27.5, -110), board, 'draftkings');
  assert.deepEqual([result.status, result.depth, result.url, result.line], ['line', 'game', 'https://sportsbook.draftkings.com/event/9', 28.5]);
});

test('a book without the market, or a leg off the board, opens the book’s home page', () => {
  assert.deepEqual(['book', 'home'], (r => [r.status, r.depth])(handoffLeg(leg('Jaylen Brown', 'REB', 6.5, 120), board, 'draftkings')));
  assert.deepEqual(['board', 'home'], (r => [r.status, r.depth])(handoffLeg(leg('Nobody', 'PTS', 9.5, -110), board, 'fanduel')));
  assert.equal(handoffLeg(leg('Bam Adebayo', 'AST', 4.5, -110), board, 'fanduel').status, 'book');
});

test('FanDuel gets one combined slip link only when every leg opens a slip selection', () => {
  const legs = [leg('Jayson Tatum', 'PTS', 27.5, -115), leg('Jaylen Brown', 'REB', 6.5, 120)];
  const { combined } = handoff(legs, { nba: board }, 'fanduel');
  assert.equal(combined, 'https://sportsbook.fanduel.com/addToBetslip?marketId[0]=42.1&selectionId[0]=7&marketId[1]=42.2&selectionId[1]=11');
  assert.equal(handoff([...legs, leg('Nobody', 'PTS', 9.5, -110)], { nba: board }, 'fanduel').combined, null);
  assert.equal(handoff(legs.slice(0, 1), { nba: board }, 'fanduel').combined, null, 'one leg uses its own link');
  assert.equal(handoff(legs, { nba: board }, 'draftkings').combined, null);
});
