/** Books a slip can be sent to, keyed the way the Odds API names them. */
export const BOOKS = {
  fanduel: { name: 'FanDuel', home: 'https://sportsbook.fanduel.com/' },
  draftkings: { name: 'DraftKings', home: 'https://sportsbook.draftkings.com/' },
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
 * status: ready | price (price moved, same selection) | line (line moved: the game page, never another selection)
 *         | book (this book has no quote) | board (the leg is not on tonight's board)
 */
export function handoffLeg(leg, board, bookKey) {
  const home = BOOKS[bookKey]?.home ?? null;
  const prop = board?.props?.find(p => p.player === leg.player && p.stat === leg.stat);
  if (!prop) return { status: 'board', depth: 'home', url: home, line: null, price: null };
  const quote = prop.books?.find(b => bookKeyOf(b.book) === bookKey);
  if (!quote) return { status: 'book', depth: 'home', url: home, line: null, price: null };

  const under = leg.direction === 'Under';
  const price = under ? quote.under_price : quote.over_price;
  const game = safeBookUrl(quote.event_link, bookKey);
  const fallback = game ? { depth: 'game', url: game } : { depth: 'home', url: home };
  if (quote.line !== leg.line) return { status: 'line', ...fallback, line: quote.line, price };

  const slip = safeBookUrl(under ? quote.under_link : quote.over_link, bookKey);
  return { status: price === leg.price ? 'ready' : 'price', ...(slip ? { depth: 'slip', url: slip } : fallback), line: quote.line, price };
}

/** The selection ids inside a book's add-to-slip link, or null when the link is not one. */
function selection(url, bookKey) {
  const params = new URL(url).searchParams;
  if (bookKey === 'fanduel') {
    const market = params.get('marketId'), selection = params.get('selectionId');
    return market && selection ? { market, selection } : null;
  }
  return params.get('outcomes') || null;
}

/**
 * Each book's own multi-add format: FanDuel's indexed addToBetslip, DraftKings' '+'-joined outcomes.
 * FanDuel reads indexed legs only on its account host (where its own shared parlays point); sportsbook.fanduel.com/addToBetslip keeps the first.
 */
const COMBINE = {
  fanduel: ids => 'https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?' + ids.map(({ market, selection }, i) =>
    `marketId[${i}]=${encodeURIComponent(market)}&selectionId[${i}]=${encodeURIComponent(selection)}`).join('&'),
  // ponytail: DraftKings' multi-outcome form is inferred from its single-outcome links; check it if DK adds only one leg.
  draftkings: ids => `${BOOKS.draftkings.home}?outcomes=${ids.map(encodeURIComponent).join('+')}`,
};

/**
 * Every leg's hand-off, plus one link that adds every leg with a slip link to the book's bet slip at once.
 * Legs without one (moved line, no quote) are left out of it, so the book never gets a selection the person did not pick.
 */
export function handoff(legs, boards, bookKey) {
  const results = legs.map(leg => handoffLeg(leg, boards?.[leg.league], bookKey));
  const slips = results.filter(r => r.depth === 'slip');
  const ids = slips.map(r => selection(r.url, bookKey)).filter(Boolean);
  const combined = ids.length === 0 || !COMBINE[bookKey] ? null : ids.length === 1 ? slips[0].url : COMBINE[bookKey](ids);
  return { legs: results, combined, linked: ids.length };
}
