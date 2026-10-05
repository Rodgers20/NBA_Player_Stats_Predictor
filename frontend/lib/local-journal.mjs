const KEY = 'court-vision-journal-v1';
const STATS = new Set(['PTS','REB','AST','FG3M','STL','BLK','PTS+REB','PTS+AST','REB+AST','PTS+REB+AST','STL+BLK']);

function read(storage) {
  const value = JSON.parse(storage.getItem(KEY) || '[]');
  if (!Array.isArray(value)) throw new Error('Saved journal data is invalid.');
  return value;
}

function write(storage, bets) {
  storage.setItem(KEY, JSON.stringify(bets));
}

export function localJournalList(storage, mode) {
  const bets = read(storage).filter(bet => bet.mode === mode);
  const settled = bets.filter(bet => ['win','loss','push'].includes(bet.result));
  const stake = settled.reduce((sum, bet) => sum + bet.stake_cents, 0);
  const profitCents = settled.reduce((sum, bet) => sum + bet.profit_cents, 0);
  return { mode, bets, summary: { count: bets.length, settled: settled.length,
    profit: profitCents / 100, roi: stake ? 100 * profitCents / stake : null,
    pending: bets.filter(bet => bet.result === 'pending').reduce((sum, bet) => sum + bet.stake_cents, 0) / 100,
    staked: stake / 100 } };
}

export function localJournalAdd(storage, entry) {
  const bets = read(storage);
  const duplicate = bets.find(bet => bet.token === entry.token);
  if (duplicate) return { id: duplicate.id };
  const player = String(entry.player || '').trim();
  const book = String(entry.book || '').trim();
  if (!player || !book) throw new Error('Player and sportsbook are required.');
  if (!['paper','real'].includes(entry.mode) || !['nba','wnba'].includes(entry.league) ||
      !STATS.has(entry.stat) || !['Over','Under'].includes(entry.side)) throw new Error('Choose a valid record type, league, stat and side.');
  if (!/^\d{4}-\d{2}-\d{2}$/.test(entry.game_date) || Number.isNaN(Date.parse(entry.game_date))) throw new Error('Enter a valid game date.');
  const line = Number(entry.line), price = Number(entry.price), stake = Number(entry.stake);
  if (!Number.isFinite(line) || line < 0 || !Number.isInteger(line * 2)) throw new Error('Line must be a nonnegative whole or half point.');
  if (!Number.isFinite(price) || (price > -100 && price < 100) || price === 0) throw new Error('Enter American odds, such as -110 or +120.');
  if (!Number.isFinite(stake) || stake <= 0) throw new Error('Stake must be greater than zero.');
  const bet = { ...entry, id: crypto.randomUUID(), player, book, line, price,
    stake_cents: Math.round(stake * 100), result: 'pending', profit_cents: null,
    created_at: new Date().toISOString() };
  delete bet.stake;
  bets.unshift(bet);
  write(storage, bets);
  return { id: bet.id };
}

export function localJournalSettle(storage, id, result) {
  if (!['win','loss','push','void','pending'].includes(result)) throw new Error('Invalid result.');
  const bets = read(storage), bet = bets.find(item => item.id === id);
  if (!bet) throw new Error('Choose a recorded bet.');
  const payout = bet.price > 0 ? bet.price / 100 : 100 / Math.abs(bet.price);
  bet.result = result;
  bet.profit_cents = result === 'win' ? Math.round(bet.stake_cents * payout) :
    result === 'loss' ? -bet.stake_cents : result === 'pending' ? null : 0;
  bet.settled_at = result === 'pending' ? null : new Date().toISOString();
  write(storage, bets);
  return { id, result };
}
