import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseScoreboard, parseSummary, withLive, statValue, findPlayer, liveBet, summaryPoll, scoreboardPoll } from './espn-live.mjs';

const status = (state, detail = '', period = 0, clock = '') => ({type: {state, shortDetail: detail, detail}, period, displayClock: clock});
const team = (id, abbreviation) => ({id, abbreviation, displayName: abbreviation, logo: `https://a.espncdn.com/${abbreviation}.png`});
const competitors = (awayScore, homeScore) => [
  {homeAway: 'home', score: String(homeScore), team: team('7', 'DEN'), linescores: [{displayValue: '30'}], record: [{type: 'total', displayValue: '3-1'}]},
  {homeAway: 'away', score: String(awayScore), team: team('3', 'NO'), linescores: [{displayValue: '25'}]},
];

test('scoreboard is keyed by event with app team codes and live clock', () => {
  const board = parseScoreboard('nba', {events: [
    {id: '401', date: '2026-10-24T23:00Z', competitions: [{competitors: competitors(55, 61)}], status: status('in', '5:21 - 3rd', 3, '5:21')},
    {id: '402', date: '2026-10-25T02:00Z', competitions: [{competitors: competitors(0, 0)}], status: status('pre', '10/24 - 10:00 PM EDT')},
  ]});
  assert.deepEqual(board['401'], {id: '401', start: '2026-10-24T23:00Z', state: 'in', detail: '5:21 - 3rd', period: 3, clock: '5:21',
    away: {abbrev: 'NOP', score: 55}, home: {abbrev: 'DEN', score: 61}});
  assert.equal(board['402'].away.score, null, 'pregame zeros are not scores');
  assert.deepEqual(parseScoreboard('nba', {}), {});
});

test('live overlay replaces snapshot score and status without mutating the game', () => {
  const game = {game_id: '401', home_team: 'DEN', away_team: 'NOP', status_state: 'pre', status_text: '7:00 PM', home_score: null};
  const live = {state: 'in', detail: 'Halftime', away: {score: 50}, home: {score: 48}};
  assert.deepEqual(withLive(game, live), {...game, status_state: 'in', status_text: 'Halftime', away_score: 50, home_score: 48});
  assert.equal(game.status_state, 'pre');
  assert.equal(withLive(game, {...live, state: 'pre'}), game, 'pregame feed never overrides the snapshot');
  assert.equal(withLive(game, undefined), game);
});

const summary = (state = 'in') => ({
  header: {id: '401', competitions: [{id: '401', status: status(state, state === 'post' ? 'Final' : '5:21 - 3rd', 3, '5:21'), competitors: competitors(55, 61)}]},
  boxscore: {
    teams: [{team: {id: '7'}, statistics: [{label: 'Rebounds', displayValue: '30'}, {label: 'Secret', displayValue: 'x'}]}],
    players: [{team: {id: '7'}, statistics: [{labels: ['MIN', 'PTS', '3PT', 'REB', 'AST', 'STL', 'BLK'], totals: ['', '61', '6-20', '30', '15', '4', '2'], athletes: [
      {starter: true, athlete: {id: '3112335', displayName: 'Nikola Jokić', shortName: 'N. Jokić', position: {abbreviation: 'C'}}, stats: ['30', '20', '2-4', '11', '9', '1', '0']},
      {didNotPlay: true, reason: 'INJURY', athlete: {id: '9', displayName: 'Bench Guy'}, stats: []},
    ]}]}],
  },
  plays: [
    {id: 'p1', text: 'Jamal Murray makes 26-foot three', scoringPlay: true, scoreValue: 3, awayScore: 52, homeScore: 61, period: {number: 3}, clock: {displayValue: '5:40'}, team: {id: '7'}, participants: [{athlete: {id: '1'}}]},
    {id: 'p2', text: 'Nikola Jokić defensive rebound', scoringPlay: false, awayScore: 52, homeScore: 61, period: {number: 3}, clock: {displayValue: '5:30'}, team: {id: '7'}, participants: [{athlete: {id: '3112335'}}]},
    {id: 'p3', text: 'Zion Williamson makes dunk', scoringPlay: true, scoreValue: 2, awayScore: 54, homeScore: 61, period: {number: 3}, clock: {displayValue: '5:21'}, team: {id: '3'}},
  ],
});

test('summary carries the box score and newest-first plays with scorer side', () => {
  const box = parseSummary('nba', summary());
  assert.equal(box.state, 'in');
  assert.equal(box.available, true);
  assert.equal(box.status, '5:21 - 3rd');
  assert.deepEqual(box.teams.map(t => t.abbrev), ['NOP', 'DEN']);
  const den = box.teams[1];
  assert.equal(den.players[0].stats.REB, '11');
  assert.equal(den.players[0].image, 'https://a.espncdn.com/i/headshots/nba/players/full/3112335.png');
  assert.deepEqual(den.team_stats, [{label: 'Rebounds', value: '30'}], 'only known team stats are shown');
  assert.deepEqual(den.leaders.map(l => [l.label, l.value]), [['PTS', 20], ['REB', 11], ['AST', 9]]);
  assert.deepEqual(box.plays.map(p => [p.id, p.side, p.scoring, p.away, p.home]),
    [['p3', 'away', true, 54, 61], ['p2', 'home', false, 52, 61], ['p1', 'home', true, 52, 61]]);
  assert.equal(box.plays[1].athlete, '3112335');
  assert.equal(parseSummary('nba', summary('pre')).available, false);
});

test('bet stats read ESPN columns, including threes made and combos', () => {
  const stats = {PTS: '20', REB: '11', AST: '9', '3PT': '2-4', STL: '1', BLK: '0'};
  assert.equal(statValue(stats, 'REB'), 11);
  assert.equal(statValue(stats, 'FG3M'), 2);
  assert.equal(statValue(stats, 'PTS+REB+AST'), 40);
  assert.equal(statValue(stats, 'STL+BLK'), 1);
  assert.equal(statValue({}, 'PTS'), null);
  assert.equal(statValue(stats, 'TOV'), null);
});

test('players are found across accents and suffixes', () => {
  const box = parseSummary('nba', summary());
  assert.equal(findPlayer(box, 'Nikola Jokic')?.name, 'Nikola Jokić');
  assert.equal(findPlayer(box, 'nikola jokić jr.')?.name, 'Nikola Jokić');
  assert.equal(findPlayer(box, 'Bench Guy'), null, 'a DNP player has no live line');
  assert.equal(findPlayer(box, 'Nobody'), null);
});

test('live bet grade: overs cash early, unders bust early, finals settle', () => {
  const bet = (side, line) => ({side, line, stat: 'REB'});
  assert.deepEqual(liveBet(bet('Over', 11.5), 8, 'in'), {value: 8, status: 'live', need: 4});
  assert.deepEqual(liveBet(bet('Over', 7.5), 8, 'in'), {value: 8, status: 'win', need: 0});
  assert.deepEqual(liveBet(bet('Over', 8), 8, 'in'), {value: 8, status: 'live', need: 1});
  assert.deepEqual(liveBet(bet('Over', 8), 8, 'post'), {value: 8, status: 'push', need: 1});
  assert.deepEqual(liveBet(bet('Over', 11.5), 8, 'post'), {value: 8, status: 'loss', need: 4});
  assert.deepEqual(liveBet(bet('Under', 7.5), 8, 'in'), {value: 8, status: 'loss', need: 0});
  assert.deepEqual(liveBet(bet('Under', 11.5), 8, 'in'), {value: 8, status: 'live', need: 0});
  assert.deepEqual(liveBet(bet('Under', 11.5), 8, 'post'), {value: 8, status: 'win', need: 0});
});

test('polling is fast only while a game is on and stops at the final', () => {
  assert.equal(summaryPoll('in'), 15_000);
  assert.equal(summaryPoll('pre'), 60_000);
  assert.equal(summaryPoll('post'), false);
  assert.equal(scoreboardPoll([{state: 'post'}, {state: 'in'}]), 20_000);
  assert.equal(scoreboardPoll([{state: 'post'}, {state: 'pre'}]), 60_000);
  assert.equal(scoreboardPoll([{state: 'post'}]), false);
  assert.equal(scoreboardPoll([]), false);
});
