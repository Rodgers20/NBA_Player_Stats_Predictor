import { isQuarterLeg } from './team-legs.mjs';

/** Books a slip can be sent to, keyed the way the Odds API names them. */
export const BOOKS = {
  fanduel: { name: 'FanDuel', home: 'https://sportsbook.fanduel.com/' },
  draftkings: { name: 'DraftKings', home: 'https://sportsbook.draftkings.com/' },
  fanatics: { name: 'Fanatics', home: 'https://sportsbook.fanatics.com/' },
};

/** A link is opened only when it is https on the book's own domain; anything else is null. */
export function safeBookUrl(url, bookKey) {
  if (typeof url !== 'string' || !bookKey) return null;
  try {
    const { protocol, hostname } = new URL(url);
    const domain = `${bookKey.toLowerCase()}.com`;
    return protocol === 'https:' && (hostname === domain || hostname.endsWith(`.${domain}`)) ? url : null;
  } catch {
    return null;
  }
}

const bookKeyOf = name => String(name ?? '').toLowerCase().replace(/[^a-z]/g, '');

/**
 * Where to send one slip leg at one book, deepest first (slip > game > home), and what the person should check.
 * status: ready | price (price moved, same selection) | line (the book hangs another number: its line is handed off, the note names it)
 *         | quarter (a 1st-quarter leg: no 1Q odds are bought, so the game page)
 *         | book (this book has no quote) | board (the leg is not on tonight's board)
 */
export function handoffLeg(leg, board, bookKey) {
  const home = BOOKS[bookKey]?.home ?? null;
  if (isQuarterLeg(leg) && !board?.props?.some(p => p.player === leg.player && p.stat === leg.stat)) {
    // No 1Q quote was bought for this player, so there is no selection to add: open the game their full-game quotes point at.
    const game = board?.props?.filter(p => p.player === leg.player).flatMap(p => p.books ?? [])
      .filter(b => bookKeyOf(b.book) === bookKey).map(b => safeBookUrl(b.event_link, bookKey)).find(Boolean);
    return { status: 'quarter', depth: game ? 'game' : 'home', url: game ?? home, line: null, price: null };
  }
  const prop = board?.props?.find(p => p.player === leg.player && p.stat === leg.stat);
  if (!prop) return { status: 'board', depth: 'home', url: home, line: null, price: null };
  const quote = prop.books?.find(b => bookKeyOf(b.book) === bookKey);
  if (!quote) return { status: 'book', depth: 'home', url: home, line: null, price: null };

  const under = leg.direction === 'Under';
  const price = under ? quote.under_price : quote.over_price;
  const game = safeBookUrl(quote.event_link, bookKey);
  const fallback = game ? { depth: 'game', url: game } : { depth: 'home', url: home };
  // The book hangs another number: hand off its own line (the note names it) rather than leave the leg out of the one-tap slip.
  const slip = safeBookUrl(under ? quote.under_link : quote.over_link, bookKey);
  const status = quote.line !== leg.line ? 'line' : price === leg.price ? 'ready' : 'price';
  return { status, ...(slip ? { depth: 'slip', url: slip, page: game } : { ...fallback, page: game }), line: quote.line, price };
}

/**
 * A copy of tonight's board with the bought 1st-quarter quotes as props, so 1Q slip legs hand off (and reprice in
 * Modify) exactly like full-game legs. They carry no model chance: 1Q has no model.
 */
export function boardWithQuarterQuotes(board, quotes) {
  const extra = (quotes ?? []).map(q => ({ player: q.player, stat: q.stat, line: q.line, books: q.books ?? [], direction: 'Over',
    model_prob: null, game_matchup: q.game_matchup }));
  return { ...(board ?? {}), props: [...(board?.props ?? []), ...extra] };
}

/** The selection ids inside a book's add-to-slip link, or null when the link is not one. */
function selection(url, bookKey) {
  const params = new URL(url).searchParams;
  if (bookKey === 'fanduel') {
    const market = params.get('marketId'), selection = params.get('selectionId');
    return market && selection ? { market, selection } : null;
  }
  return bookKey === 'draftkings' ? params.get('outcomes') || null : null;
}

/**
 * Each book's multi-add link, copied from the ones Gambly's share pages send (live, Nov 2026):
 * FanDuel's stateless addToBetslip with every `marketId[i]=..&selectionId[i]=..` pair percent-encoded whole (no state site),
 * DraftKings' event page with the outcomes comma-joined (legs from other games ride along on the first one's page).
 */
const COMBINE = {
  fanduel: ids => 'https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?' + ids.map(({ market, selection }, i) =>
    `marketId%5b${i}%5d%3d${encodeURIComponent(market)}%26selectionId%5b${i}%5d%3d${encodeURIComponent(selection)}`).join('&'),
  draftkings: (ids, page) => `${page && /\/event\/\d+$/.test(page) ? page : BOOKS.draftkings.home}?outcomes=${ids.map(encodeURIComponent).join(',')}`,
};

/**
 * Every leg's hand-off, plus one link that adds every leg with a slip link to the book's bet slip at once.
 * Legs without one (no quote, or a quote with no link) are left out of it.
 */
export function handoff(legs, boards, bookKey) {
  const results = legs.map(leg => handoffLeg(leg, boards?.[leg.league], bookKey));
  const slips = results.filter(r => r.depth === 'slip');
  if (slips.length === 1) return { legs: results, combined: slips[0].url, linked: 1 };
  // ponytail: Fanatics has no known multi-leg format, so several linked legs open one by one; add it to COMBINE once seen.
  const ids = COMBINE[bookKey] ? slips.map(r => selection(r.url, bookKey)).filter(Boolean) : [];
  return { legs: results, combined: ids.length ? COMBINE[bookKey](ids, slips[0]?.page) : null, linked: ids.length };
}
