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

test('FanDuel: one link adds every linked leg to the bet slip, in the indexed form FanDuel shares parlays with', () => {
  const legs = [leg('Jayson Tatum', 'PTS', 27.5, -115), leg('Jaylen Brown', 'REB', 6.5, 120)];
  const all = handoff(legs, { nba: board }, 'fanduel');
  assert.equal(all.combined, 'https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?marketId[0]=42.1&selectionId[0]=7&marketId[1]=42.2&selectionId[1]=11');
  assert.equal(all.linked, 2);
  const partial = handoff([...legs, leg('Nobody', 'PTS', 9.5, -110)], { nba: board }, 'fanduel');
  assert.equal(partial.combined, all.combined, 'legs without a slip link are left for the person to add');
  assert.equal(partial.linked, 2);
  assert.equal(handoff(legs.slice(0, 1), { nba: board }, 'fanduel').combined, fdSlip('42.1', '7'));
  assert.deepEqual(handoff(legs, { nba: board }, 'draftkings'), { ...handoff(legs, { nba: board }, 'draftkings'), combined: null, linked: 0 });
});

test('DraftKings: one link carries every linked outcome id', () => {
  const dk = id => `https://sportsbook.draftkings.com/?outcomes=${encodeURIComponent(id)}`;
  const dkBoard = { props: [
    { player: 'Jayson Tatum', stat: 'PTS', books: [{ book: 'DraftKings', line: 26.5, over_price: -108, under_price: -118, over_link: dk('0QA382972274#2337722395_13L42648Q11742021793Q20') }] },
    { player: 'Michael Porter Jr', stat: 'PTS', books: [{ book: 'DraftKings', line: 22.5, over_price: -103, under_price: -124, under_link: dk('0QA382971620#2337695081_13L42648Q12094929165Q20') }] },
  ] };
  const legs = [leg('Jayson Tatum', 'PTS', 26.5, -108), leg('Michael Porter Jr', 'PTS', 22.5, -124, 'Under')];
  const { combined, linked } = handoff(legs, { nba: dkBoard }, 'draftkings');
  assert.equal(linked, 2);
  assert.equal(combined, 'https://sportsbook.draftkings.com/?outcomes=0QA382972274%232337722395_13L42648Q11742021793Q20+0QA382971620%232337695081_13L42648Q12094929165Q20');
});

test('Fanatics: no quote opens its home page; a single slip link is used as is, several wait for a known parlay format', () => {
  const none = handoff([leg('Jayson Tatum', 'PTS', 27.5, -115)], { nba: board }, 'fanatics');
  assert.deepEqual([none.legs[0].status, none.legs[0].url, none.combined], ['book', 'https://sportsbook.fanatics.com/', null]);
  const fx = (id) => `https://sportsbook.fanatics.com/addToBetslip?selection=${id}`;
  const fxBoard = { props: [
    { player: 'Jayson Tatum', stat: 'PTS', books: [{ book: 'Fanatics', line: 27.5, over_price: -115, over_link: fx(1) }] },
    { player: 'Jaylen Brown', stat: 'REB', books: [{ book: 'Fanatics', line: 6.5, over_price: 120, over_link: fx(2) }] },
  ] };
  const one = handoff([leg('Jayson Tatum', 'PTS', 27.5, -115)], { nba: fxBoard }, 'fanatics');
  assert.deepEqual([one.combined, one.linked], [fx(1), 1]);
  const two = handoff([leg('Jayson Tatum', 'PTS', 27.5, -115), leg('Jaylen Brown', 'REB', 6.5, 120)], { nba: fxBoard }, 'fanatics');
  assert.deepEqual([two.combined, two.linked], [null, 0]);
});

test('a 1st-quarter leg opens the player’s game page at the book (no 1Q odds are bought), never a selection', () => {
  const q1 = handoffLeg(leg('Jayson Tatum', 'PTS_1Q', 7.5, -110), board, 'fanduel');
  assert.deepEqual([q1.status, q1.depth, q1.url], ['quarter', 'game', FD_GAME]);
  const unknown = handoffLeg(leg('Nobody', 'PTS_1Q', 3.5, -110), board, 'fanduel');
  assert.deepEqual([unknown.status, unknown.depth, unknown.url], ['quarter', 'home', 'https://sportsbook.fanduel.com/']);
  assert.equal(handoff([leg('Jayson Tatum', 'PTS_1Q', 7.5, -110)], { nba: board }, 'fanduel').combined, null);
});

test('FanDuel with a state: every leg goes to that state’s site in the indexed form, exactly as Gambly’s working links do', () => {
  const legs = [leg('Jayson Tatum', 'PTS', 27.5, -115), leg('Jaylen Brown', 'REB', 6.5, 120)];
  assert.equal(handoff(legs, { nba: board }, 'fanduel', 'va').combined,
    'https://va.sportsbook.fanduel.com/addToBetslip?marketId[0]=42.1&selectionId[0]=7&marketId[1]=42.2&selectionId[1]=11');
  assert.equal(handoff(legs.slice(0, 1), { nba: board }, 'fanduel', 'va').combined, 'https://va.sportsbook.fanduel.com/addToBetslip?marketId[0]=42.1&selectionId[0]=7');
  assert.equal(handoff(legs, { nba: board }, 'fanduel', 'evil.com/x').combined, handoff(legs, { nba: board }, 'fanduel').combined, 'an unknown state is ignored');
});
