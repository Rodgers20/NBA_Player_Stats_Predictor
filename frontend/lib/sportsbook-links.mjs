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

/**
 * Every leg's hand-off, plus one FanDuel link that adds them all to the slip when every leg has its own slip link.
 * ponytail: combined format is FanDuel's indexed addToBetslip; unverified until real links arrive, per-leg links always shown.
 */
export function handoff(legs, boards, bookKey) {
  const results = legs.map(leg => handoffLeg(leg, boards?.[leg.league], bookKey));
  let combined = null;
  if (bookKey === 'fanduel' && results.length > 1 && results.every(r => r.depth === 'slip')) {
    const ids = results.map(r => new URL(r.url).searchParams);
    if (ids.every(p => p.get('marketId') && p.get('selectionId'))) {
      combined = `${BOOKS.fanduel.home}addToBetslip?` + ids.map((p, i) =>
        `marketId[${i}]=${encodeURIComponent(p.get('marketId'))}&selectionId[${i}]=${encodeURIComponent(p.get('selectionId'))}`).join('&');
    }
  }
  return { legs: results, combined };
}
