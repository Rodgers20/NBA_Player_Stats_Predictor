const sidePrice = prop => (/^under$/i.test(prop.direction) ? 'under_price' : 'over_price');

/** Profit per unit staked at an American price. */
const profitPerUnit = price => (price > 0 ? price / 100 : 100 / -price);

/**
 * The pick as the chosen sportsbooks would sell it, or null when none of them quotes this line and side.
 * Shows the best chosen price and recomputes EV from the model probability. A whole-number line can push, and the
 * push share is not carried on the pick, so a new price there drops the EV rather than report a wrong one.
 * Stars and edge belong to the original price, so they are cleared when the price changes.
 */
export function forBooks(prop, books) {
  if (!books?.length) return prop;
  const field = sidePrice(prop);
  const chosen = new Set(books);
  const quotes = (prop.books ?? []).filter(quote => chosen.has(quote.book) && quote.line === prop.line && quote[field] != null);
  if (!quotes.length) return null;
  const best = quotes.reduce((top, quote) => (quote[field] > top[field] ? quote : top));
  if (best[field] === prop.price) return { ...prop, book: best.book };
  const exact = prop.model_prob != null && !Number.isInteger(prop.line);
  const ev = exact ? prop.model_prob * profitPerUnit(best[field]) - (1 - prop.model_prob) : null;
  return { ...prop, price: best[field], book: best.book, ev, stars: null, edge_pct: null,
    pick_type: ev != null && ev > 0 ? 'pick' : 'lean' };
}

/** +EV only, stat and player pickers, and the chosen sportsbooks, applied in one pass. Empty selections mean "any". */
export function applyBoardFilters(props, { evOnly = false, stats = [], players = [], books = [] } = {}) {
  const wantedStats = new Set(stats);
  const wantedPlayers = new Set(players);
  return props
    .filter(prop => (!wantedStats.size || wantedStats.has(prop.stat)) && (!wantedPlayers.size || wantedPlayers.has(prop.player)))
    .map(prop => forBooks(prop, books))
    .filter(prop => prop && (!evOnly || prop.pick_type === 'pick'));
}

const LINE_MARKETS = new Set(['ML', 'SPREAD', 'TOTAL']);

/** The comma-separated stat selection as a list; empty means every stat. */
export const selectedStats = stat => (stat ? stat.split(',') : []);

/**
 * Clicking a stat chip: toggles it in the selection, "All" ('') clears it, and a whole-game line market
 * (which is a different board) replaces the player stats and toggles off on its own.
 */
export function toggleStat(current, value) {
  if (value === '') return '';
  const chosen = selectedStats(current);
  if (LINE_MARKETS.has(value)) return chosen.length === 1 && chosen[0] === value ? '' : value;
  const players = chosen.filter(stat => !LINE_MARKETS.has(stat));
  const next = players.includes(value) ? players.filter(stat => stat !== value) : [...players, value];
  return next.join(',');
}

const sortedUnique = values => [...new Set(values)].sort((a, b) => a.localeCompare(b));

/** Every sportsbook that quotes at least one loaded pick. */
export const availableBooks = props => sortedUnique(props.flatMap(prop => (prop.books ?? []).map(quote => quote.book)));

/** Every player on the loaded board. */
export const availablePlayers = props => sortedUnique(props.map(prop => prop.player));
