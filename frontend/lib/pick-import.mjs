import { ASSUMED_PRICE } from './team-legs.mjs';

/* Pasted picks ("Brunson o27.5 pts, SGA 30+ points") → slip legs, matched only against tonight's board. */

const STAT_ALIASES = [
  ['PTS+REB+AST', ['pts+reb+ast', 'p+r+a', 'pra', 'points rebounds assists', 'points + rebounds + assists']],
  ['PTS+REB', ['pts+reb', 'p+r', 'pr', 'points + rebounds', 'points rebounds']],
  ['PTS+AST', ['pts+ast', 'p+a', 'pa', 'points + assists', 'points assists']],
  ['REB+AST', ['reb+ast', 'r+a', 'ra', 'rebounds + assists', 'rebounds assists']],
  ['FG3M', ['threes', 'three pointers', '3pm', '3pt', '3s', 'treys']],
  ['PTS', ['points', 'pts', 'point']],
  ['REB', ['rebounds', 'rebs', 'reb', 'boards']],
  ['AST', ['assists', 'asts', 'ast', 'dimes']],
  ['BLK', ['blocks', 'blk', 'blks']],
  ['STL', ['steals', 'stl', 'stls']],
];
const ALIASES = STAT_ALIASES.flatMap(([stat, words]) => words.map(word => [word, stat])).sort((a, b) => b[0].length - a[0].length);
const escape = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const STAT_RE = new RegExp(`(?<![a-z0-9])(${ALIASES.map(([w]) => escape(w)).join('|')})(?![a-z0-9])`);
const STAT_OF = Object.fromEntries(ALIASES);

// ponytail: a handful of common nicknames; initials, first and last names are matched without a list.
const NICKNAMES = { ant: 'anthony edwards', wemby: 'victor wembanyama', joker: 'nikola jokic', dame: 'damian lillard',
  steph: 'stephen curry', book: 'devin booker', spida: 'donovan mitchell', bron: 'lebron james', cp3: 'chris paul' };

const ORDER = ['PTS', 'REB', 'AST'];
const canonStat = stat => stat.split('+').sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b)).join('+');
const normName = name => name.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z0-9\s-]/g, '').trim();
const nameTokens = name => normName(name).split(/[\s-]+/).filter(Boolean);

/** One pick per line, comma or semicolon; list markers stripped; lines that are not a pick dropped. */
export function splitPicks(text) {
  return String(text ?? '').split(/[\n,;]+/)
    .map(part => part.replace(/^\s*(?:\d+[.)]|[•·*-])\s*/, '').trim())
    .filter(part => parsePick(part) !== null);
}

/** "SGA u 41.5 P+R+A (-110)" → { query: 'sga', direction: 'Under', line: 41.5, stat: 'PTS+REB+AST' }; null if not a pick. */
export function parsePick(text) {
  let rest = String(text ?? '').toLowerCase().replace(/[’']/g, '').replace(/\(\s*[+-]?\d+\s*\)/g, ' ');
  const statMatch = rest.match(STAT_RE);
  if (!statMatch) return null;
  rest = rest.replace(statMatch[0], ' ');

  let direction = 'Over', line = null, assumedDirection = false, m;
  if ((m = rest.match(/(?<![a-z])(over|under|o|u)\s*(\d+(?:\.\d+)?)/))) {
    direction = m[1].startsWith('u') ? 'Under' : 'Over'; line = Number(m[2]);
  } else if ((m = rest.match(/(\d+(?:\.\d+)?)\s*\+/))) {
    line = Number(m[1]) - 0.5;
  } else if ((m = rest.match(/(\d+(?:\.\d+)?)/))) {
    line = Number(m[1]); assumedDirection = true;
  }
  if (line === null) return null;
  const query = rest.replace(m[0], ' ').replace(/[^a-z0-9\s-]/g, ' ').replace(/\s+/g, ' ').trim();
  return query ? { query, direction, line, stat: STAT_OF[statMatch[1]], assumedDirection } : null;
}

/** Board players the query can mean: full name, nickname, initials, or every query word is one of the name's words. */
function matchPlayers(query, names) {
  const q = normName(query);
  const exact = names.filter(n => normName(n) === q || NICKNAMES[q] === normName(n));
  if (exact.length) return exact;
  const words = q.split(/[\s-]+/);
  return names.filter(n => {
    const tokens = nameTokens(n);
    return tokens.map(t => t[0]).join('') === q || words.every(w => tokens.includes(w));
  });
}

function legFor(prop, league, date, direction, quote) {
  const price = quote ? (direction === 'Under' ? quote.under_price : quote.over_price) : null;
  const line = quote ? quote.line : prop.line;
  const modelSide = prop.direction === direction && prop.line === line && Number.isFinite(prop.model_prob);
  return { league, date, player: prop.player, stat: prop.stat, direction, line,
    price: Number.isFinite(price) ? price : ASSUMED_PRICE, ...(Number.isFinite(price) ? {} : { assumedPrice: true }),
    probability: modelSide ? prop.model_prob : null, matchup: prop.game_matchup, team: prop.team, addedAt: new Date().toISOString() };
}

/**
 * Each pasted pick → { text, status, leg?, requested?, candidates? }.
 * status: exact (ready) | line (a different line is offered for approval) | ambiguous | player | market
 */
export function importPicks(text, boards) {
  const entries = Object.entries(boards ?? {}).flatMap(([league, board]) =>
    (board?.props ?? []).map(prop => ({ league, date: board.target_date, prop })));
  const names = [...new Set(entries.map(e => e.prop.player))];

  return splitPicks(text).map(pickText => {
    const pick = parsePick(pickText);
    const players = matchPlayers(pick.query, names);
    if (!players.length) return { text: pickText, status: 'player' };
    if (players.length > 1) return { text: pickText, status: 'ambiguous', candidates: players };

    const entry = entries.find(e => e.prop.player === players[0] && canonStat(e.prop.stat) === pick.stat);
    if (!entry) return { text: pickText, status: 'market' };
    const { league, date, prop } = entry;
    const priceOf = b => (pick.direction === 'Under' ? b.under_price : b.over_price);
    const quotes = (prop.books ?? []).filter(b => Number.isFinite(b.line) && Number.isFinite(priceOf(b)));
    const best = list => list.reduce((a, b) => (priceOf(b) > priceOf(a) ? b : a), list[0]);

    const atLine = quotes.filter(b => b.line === pick.line);
    if (atLine.length) return { text: pickText, status: 'exact', leg: legFor(prop, league, date, pick.direction, best(atLine)) };
    if (!quotes.length && prop.line === pick.line) return { text: pickText, status: 'exact', leg: legFor(prop, league, date, pick.direction, null) };
    const gap = Math.min(...quotes.map(q => Math.abs(q.line - pick.line)));
    const nearest = quotes.length ? best(quotes.filter(b => Math.abs(b.line - pick.line) === gap)) : null;
    return { text: pickText, status: 'line', requested: pick.line, leg: legFor(prop, league, date, pick.direction, nearest) };
  });
}
