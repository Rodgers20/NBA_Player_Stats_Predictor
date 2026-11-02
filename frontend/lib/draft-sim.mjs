// NBA draft simulator: project the season, run the lottery, then draw the draft from the consensus board.
// Pure and seedable so the page and the tests share one implementation.

export const ROUND = 30;
export const PICKS = ROUND * 2;
const LOTTERY = 16;          // picks 1-16 are decided by lottery balls
const RELEGATED = 3;         // the three worst records are guaranteed a top-12 pick
const RELEGATED_CEILING = 12;
const GAMES = 82;
const PRIOR_GAMES = 25;      // a team's net rating counts as much as this many games of the league average
const RATING_SD = 2.0;       // points per game of real uncertainty in a team's strength
const WIN_PCT_PER_POINT = 0.033;
const BOARD_SD_BASE = 0.8;   // spread of a prospect's draft slot around his board rank: base + per-rank part
const BOARD_SD_PER_RANK = 0.22;

/** mulberry32: a small seedable generator returning floats in [0, 1). */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function normal(rand) {
  return Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
}

/** Team slugs from worst projected final record to best, one random season. */
export function projectOrder(teams, rand) {
  const scored = teams.map(t => {
    const played = t.w + t.l;
    const rating = t.power * (played / (played + PRIOR_GAMES)) + RATING_SD * normal(rand);
    const p = Math.min(0.9, Math.max(0.1, 0.5 + WIN_PCT_PER_POINT * rating));
    const left = Math.max(0, GAMES - played);
    const wins = t.w + p * left + Math.sqrt(left * p * (1 - p)) * normal(rand);
    return { slug: t.slug, wins: wins + rand() * 1e-6 };
  });
  return scored.sort((a, b) => a.wins - b.wins).map(s => s.slug);
}

function weightedPick(slugs, weight, rand) {
  const total = slugs.reduce((s, x) => s + weight(x), 0);
  let r = rand() * total;
  for (const x of slugs) {
    r -= weight(x);
    if (r <= 0) return x;
  }
  return slugs[slugs.length - 1];
}

/** Draw the 16 lottery picks. `order` is worst record first; balls are per seed; restricted maps a team to the number of top picks it cannot win. */
export function drawLottery(order, balls, restricted, rand) {
  const seeds = order.slice(0, LOTTERY);
  const ballsOf = Object.fromEntries(seeds.map((s, i) => [s, balls[i]]));
  for (let attempt = 0; attempt < 200; attempt++) {
    const left = [...seeds];
    const picks = [];
    for (let pick = 1; pick <= LOTTERY; pick++) {
      const open = left.filter(s => !(restricted[s] >= pick));
      const chosen = weightedPick(open.length ? open : left, s => ballsOf[s], rand);
      picks.push(chosen);
      left.splice(left.indexOf(chosen), 1);
    }
    const protectedOk = seeds.slice(0, RELEGATED).every(s => picks.indexOf(s) < RELEGATED_CEILING);
    if (protectedOk) return picks;
  }
  return seeds;
}

/** Original owner of every pick 1-60: the lottery, then record order for 17-30, then reverse record order for round two. */
export function pickOrigins(order, lottery) {
  const first = [...lottery, ...order.slice(LOTTERY)];
  return [...first, ...order];                 // round two: worst record picks first
}

/** Prospect indexes in the order teams would call them: board rank plus noise that widens down the board. */
export function drawBoard(board, rand) {
  return board
    .map((p, i) => ({ i, key: p.rank + (BOARD_SD_BASE + BOARD_SD_PER_RANK * p.rank) * normal(rand) }))
    .sort((a, b) => a.key - b.key)
    .map(x => x.i);
}

/** One full mock draft: 60 picks with the original team, who owns the pick now, and the prospect index (or null). */
export function simulate(data, rand) {
  const order = projectOrder(data.teams, rand);
  const origins = pickOrigins(order, drawLottery(order, data.lottery.balls, data.lottery.restricted, rand));
  const players = drawBoard(data.board, rand);
  return origins.map((origin, k) => {
    const round = k < ROUND ? 'r1' : 'r2';
    return { pick: k + 1, origin, owner: data.owners[origin]?.[round] ?? origin, prospect: players[k] ?? null };
  });
}

/** Run many drafts. Returns per-pick owner and prospect odds plus a profile for every prospect. */
export function monteCarlo(data, runs = 3000, seed = 7) {
  const rand = rng(seed);
  const owners = Array.from({ length: PICKS }, () => ({}));
  const origins = Array.from({ length: PICKS }, () => ({}));
  const players = Array.from({ length: PICKS }, () => ({}));
  const slots = data.board.map(() => new Array(PICKS + 1).fill(0));      // index 0 = not drafted
  for (let r = 0; r < runs; r++) {
    const draft = simulate(data, rand);
    const taken = new Set();
    for (const p of draft) {
      const k = p.pick - 1;
      owners[k][p.owner] = (owners[k][p.owner] ?? 0) + 1;
      origins[k][p.origin] = (origins[k][p.origin] ?? 0) + 1;
      if (p.prospect != null) {
        players[k][p.prospect] = (players[k][p.prospect] ?? 0) + 1;
        slots[p.prospect][p.pick] += 1;
        taken.add(p.prospect);
      }
    }
    data.board.forEach((_, i) => { if (!taken.has(i)) slots[i][0] += 1; });
  }
  const odds = table => table.map(t => Object.entries(t).map(([key, n]) => ({ key, p: n / runs })).sort((a, b) => b.p - a.p));
  const profile = slots.map(row => {
    const p = n => n / runs;
    const drafted = row.slice(1);
    const sum = (from, to) => drafted.slice(from - 1, to).reduce((s, n) => s + n, 0);
    const taken = runs - row[0];
    const expected = taken ? drafted.reduce((s, n, k) => s + n * (k + 1), 0) / taken : null;
    return { top3: p(sum(1, 3)), top10: p(sum(1, 10)), lottery: p(sum(1, 14)), first: p(sum(1, ROUND)), drafted: p(taken), expected };
  });
  return { runs, owners: odds(owners), origins: odds(origins), players: odds(players), profile };
}
