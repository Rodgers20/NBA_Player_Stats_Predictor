// Pure filter / sort / paging logic for the props board. Prop rows follow
// soccer/docs/props-contract.md (`props/board`).

export const MARKETS = ["anytime_goal", "shots_on_target", "assist"];
const MARKET_URL = { anytime_goal: "goal", shots_on_target: "sot", assist: "assist" };
const URL_MARKET = Object.fromEntries(Object.entries(MARKET_URL).map(([k, v]) => [v, k]));
export const MARKET_LABEL = { anytime_goal: "Anytime scorer", shots_on_target: "Shots on target", assist: "Assist" };
const CONF_RANK = { low: 0, medium: 1, high: 2 };

export const DEFAULT_FILTERS = Object.freeze({
  market: "all", line: null, venue: "all", conf: "all", minProb: 0, teams: [], fixture: "all", sort: "prob",
});
export const PAGE_SIZE = 30;

const first = (v) => (Array.isArray(v) ? v[0] : v);
const all = (v) => (v === undefined ? [] : Array.isArray(v) ? v : [v]);

/** Read filters from URLSearchParams or a Next `searchParams` object; bad values fall back to defaults. */
export function parseFilters(input) {
  const get = (k) => (input instanceof URLSearchParams ? input.get(k) ?? undefined : first(input?.[k]));
  const getAll = (k) => (input instanceof URLSearchParams ? input.getAll(k) : all(input?.[k]));
  const market = URL_MARKET[get("m")] ?? "all";
  const line = Number(get("line"));
  const p = Math.round(Number(get("p")));
  const venue = get("v");
  const conf = get("c");
  const sort = get("s");
  return {
    market,
    line: line === 0.5 || line === 1.5 ? line : null,
    venue: venue === "home" || venue === "away" ? venue : "all",
    conf: conf === "medium" || conf === "high" ? conf : "all",
    minProb: Number.isFinite(p) ? Math.min(90, Math.max(0, p)) : 0,
    teams: getAll("t").filter(Boolean),
    fixture: get("f") || "all",
    sort: sort === "expected" || sort === "avg" ? sort : "prob",
  };
}

/** Only non-default filters, so a clean board has a clean URL. */
export function toQuery(filters) {
  const q = new URLSearchParams();
  if (filters.market !== "all") q.set("m", MARKET_URL[filters.market]);
  if (filters.line !== null) q.set("line", String(filters.line));
  if (filters.venue !== "all") q.set("v", filters.venue);
  if (filters.conf !== "all") q.set("c", filters.conf);
  if (filters.minProb > 0) q.set("p", String(filters.minProb));
  for (const t of filters.teams) q.append("t", t);
  if (filters.fixture !== "all") q.set("f", filters.fixture);
  if (filters.sort !== "prob") q.set("s", filters.sort);
  return q.toString();
}

function keep(prop, f, skipMarket) {
  if (!skipMarket && f.market !== "all" && prop.market !== f.market) return false;
  if (f.line !== null && prop.market === "shots_on_target" && prop.line !== f.line) return false;
  if (f.venue !== "all" && prop.is_home !== (f.venue === "home")) return false;
  if (f.conf !== "all" && (CONF_RANK[prop.confidence] ?? 0) < CONF_RANK[f.conf]) return false;
  if (prop.prob * 100 < f.minProb) return false;
  if (f.teams.length && !f.teams.includes(prop.team)) return false;
  return true;
}

const sortValue = { prob: (p) => p.prob, expected: (p) => p.expected ?? 0, avg: (p) => p.per_game ?? 0 };

/** Fixtures with filtered, sorted props; fixtures left with no rows are dropped. Input is not mutated. */
export function applyFilters(fixtures, filters, { skipMarket = false } = {}) {
  const by = sortValue[filters.sort];
  return fixtures
    .filter((fx) => filters.fixture === "all" || fx.fixture_key === filters.fixture)
    .map((fx) => ({
      ...fx,
      props: fx.props
        .filter((p) => keep(p, filters, skipMarket))
        .sort((a, b) => by(b) - by(a) || a.player.localeCompare(b.player)),
    }))
    .filter((fx) => fx.props.length > 0);
}

/** Rows per market under every filter except the market itself (for tab badges). */
export function marketCounts(fixtures, filters) {
  const counts = { all: 0, anytime_goal: 0, shots_on_target: 0, assist: 0 };
  for (const fx of applyFilters(fixtures, filters, { skipMarket: true })) {
    for (const p of fx.props) {
      counts.all += 1;
      if (p.market in counts) counts[p.market] += 1;
    }
  }
  return counts;
}

/** Keep the first `limit` rows across fixtures (for "load more"). */
export function takeRows(fixtures, limit) {
  const out = [];
  let left = limit;
  for (const fx of fixtures) {
    if (left <= 0) break;
    const props = fx.props.slice(0, left);
    left -= props.length;
    out.push({ ...fx, props });
  }
  return out;
}

export const totalRows = (fixtures) => fixtures.reduce((n, fx) => n + fx.props.length, 0);

/** Distinct team names on the board, sorted. */
export const teamsOf = (fixtures) =>
  [...new Set(fixtures.flatMap((fx) => [fx.home_team, fx.away_team]))].sort((a, b) => a.localeCompare(b));

/** hot >= 50%, warm >= 30%, else cool. */
export const probTone = (prob) => (prob >= 0.5 ? "hot" : prob >= 0.3 ? "warm" : "cool");
