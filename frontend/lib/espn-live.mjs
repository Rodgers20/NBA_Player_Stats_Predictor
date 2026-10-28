/**
 * ESPN's public basketball feeds, read straight from the browser (they send CORS `*`).
 * The hosted site is a frozen snapshot, so live scores, plays and bet progress can only come from here.
 * The box score mirrors utils/basketball_boxscore.py, which still builds the snapshot copy.
 */
export const ESPN = 'https://site.api.espn.com/apis/site/v2/sports/basketball';

const ALIASES = {
  nba: {PHO: 'PHX', NY: 'NYK', GS: 'GSW', NO: 'NOP', SA: 'SAS', UTAH: 'UTA', WSH: 'WAS', NJ: 'BKN'},
  wnba: {NY: 'NYL', GS: 'GSV', GSW: 'GSV', LA: 'LAS', LV: 'LVA', WSH: 'WAS', POR: 'PDX'},
};
const TEAM_STATS = ['FG', 'Field Goal %', '3PT', 'Three Point %', 'FT', 'Free Throw %', 'Rebounds', 'Offensive Rebounds',
  'Assists', 'Steals', 'Blocks', 'Total Turnovers', 'Points in Paint', 'Fast Break Points', 'Largest Lead'];
const LEADERS = [['PTS', 'Points'], ['REB', 'Rebounds'], ['AST', 'Assists']];

const normalizeAbbrev = (league, abbrev = '') => ALIASES[league]?.[abbrev] ?? abbrev;
const int = value => { const n = parseInt(value, 10); return Number.isFinite(n) ? n : 0; };
const espnUrl = value => typeof value === 'string' && /^https:\/\/([a-z0-9-]+\.)*(espn\.com|espncdn\.com)\//i.test(value) ? value : null;

function liveStatus(status = {}) {
  return {state: status.type?.state ?? 'pre', detail: status.type?.shortDetail || status.type?.detail || '',
    period: status.period ?? null, clock: status.displayClock ?? null};
}

/** Scoreboard for one date → {eventId: live game}. Scores stay null before tip-off. */
export function parseScoreboard(league, json) {
  const out = {};
  for (const event of json?.events ?? []) {
    const status = liveStatus(event.status ?? event.competitions?.[0]?.status);
    const sides = Object.fromEntries((event.competitions?.[0]?.competitors ?? []).map(c => [c.homeAway, {
      abbrev: normalizeAbbrev(league, c.team?.abbreviation), score: status.state === 'pre' ? null : int(c.score)}]));
    if (sides.away && sides.home) out[String(event.id)] = {id: String(event.id), start: event.date ?? null, ...status, away: sides.away, home: sides.home};
  }
  return out;
}

/** Snapshot game with the live score and clock laid over it; untouched before tip-off. */
export function withLive(game, live) {
  if (!live || live.state === 'pre') return game;
  return {...game, status_state: live.state, status_text: live.detail, away_score: live.away.score, home_score: live.home.score};
}

function headshot(league, athlete) {
  const id = String(athlete.id ?? '');
  return espnUrl(athlete.headshot?.href) ?? (/^\d+$/.test(id) ? `https://a.espncdn.com/i/headshots/${league}/players/full/${id}.png` : null);
}

function player(league, entry, labels) {
  const athlete = entry.athlete ?? {};
  return {id: String(athlete.id ?? ''), name: athlete.displayName ?? '', image: headshot(league, athlete),
    short: athlete.shortName ?? athlete.displayName ?? '', position: athlete.position?.abbreviation ?? '', jersey: athlete.jersey ?? '',
    starter: Boolean(entry.starter), dnp: Boolean(entry.didNotPlay), reason: entry.reason || null,
    stats: Object.fromEntries(labels.map((label, i) => [label, (entry.stats ?? [])[i]]).filter(([, v]) => v !== undefined))};
}

function leaders(players) {
  return LEADERS.flatMap(([label, title]) => {
    const played = players.filter(p => !p.dnp && label in p.stats);
    if (!played.length) return [];
    const best = played.reduce((a, b) => int(b.stats[label]) > int(a.stats[label]) ? b : a);
    return [{label, title, name: best.short, value: int(best.stats[label]), id: best.id, image: best.image}];
  });
}

function plays(json, sideByTeam) {
  return (json.plays ?? []).map(play => ({
    id: String(play.id), text: play.text ?? '', period: play.period?.number ?? null, clock: play.clock?.displayValue ?? '',
    side: sideByTeam[play.team?.id] ?? null, scoring: Boolean(play.scoringPlay), points: play.scoreValue ?? 0,
    away: play.awayScore ?? null, home: play.homeScore ?? null, athlete: play.participants?.[0]?.athlete?.id ?? null,
  })).reverse();
}

/** Game summary → the BoxScore shape the app already renders, plus newest-first play-by-play. */
export function parseSummary(league, json) {
  const competition = json?.header?.competitions?.[0] ?? {};
  const status = liveStatus(competition.status);
  const groups = Object.fromEntries((json?.boxscore?.players ?? []).map(g => [g.team?.id, g.statistics?.[0] ?? {}]));
  const totals = Object.fromEntries((json?.boxscore?.teams ?? []).map(t => [t.team?.id,
    Object.fromEntries((t.statistics ?? []).map(s => [s.label, s.displayValue]))]));
  const sideByTeam = {};
  const teams = ['away', 'home'].flatMap(side => {
    const competitor = (competition.competitors ?? []).find(c => c.homeAway === side);
    if (!competitor) return [];
    const team = competitor.team ?? {};
    sideByTeam[team.id] = side;
    const group = groups[team.id] ?? {};
    const labels = group.labels ?? [];
    const players = (group.athletes ?? []).map(a => player(league, a, labels));
    const stats = totals[team.id] ?? {};
    return [{side, abbrev: normalizeAbbrev(league, team.abbreviation), name: team.displayName ?? '', logo: team.logo ?? team.logos?.[0]?.href ?? null,
      color: team.color ?? null, score: status.state === 'pre' ? null : int(competitor.score), winner: Boolean(competitor.winner),
      record: (competitor.record ?? []).find(r => r.type === 'total')?.displayValue ?? null,
      linescore: (competitor.linescores ?? []).map(p => int(p.displayValue)), columns: labels, players,
      totals: Object.fromEntries(labels.map((label, i) => [label, (group.totals ?? [])[i] ?? ''])),
      team_stats: TEAM_STATS.filter(label => label in stats).map(label => ({label, value: stats[label]})), leaders: leaders(players)}];
  });
  return {game_id: String(competition.id ?? json?.header?.id ?? ''), ...status, status: status.detail, teams,
    available: status.state !== 'pre' && teams.some(t => t.players.length), plays: plays(json ?? {}, sideByTeam)};
}

/** A journal stat (PTS, FG3M, PTS+REB…) from ESPN box-score columns; null when a column is missing. */
export function statValue(stats, stat) {
  let total = 0;
  for (const part of stat.split('+')) {
    const raw = part === 'FG3M' ? stats['3PT']?.split('-')[0] : stats[part];
    const n = parseInt(raw, 10);
    if (!Number.isFinite(n)) return null;
    total += n;
  }
  return total;
}

export const nameKey = name => String(name).normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase()
  .replace(/[.'’-]/g, '').replace(/\b(jr|sr|ii|iii|iv)\b/g, '').replace(/\s+/g, ' ').trim();

/** The player's live line in a box score, or null when absent or did not play. */
export function findPlayer(box, name) {
  const key = nameKey(name);
  for (const team of box?.teams ?? []) {
    const found = team.players.find(p => !p.dnp && nameKey(p.name) === key);
    if (found) return found;
  }
  return null;
}

/** Where a bet stands now. Overs can cash and unders can bust before the final; `need` is what an over still needs. */
export function liveBet(bet, value, state) {
  const over = bet.side !== 'Under';
  const need = over ? Math.max(0, Math.floor(bet.line) + 1 - value) : 0;
  const final = state === 'post';
  const status = value === bet.line && final ? 'push'
    : over ? (value > bet.line ? 'win' : final ? 'loss' : 'live')
      : (value > bet.line ? 'loss' : final ? 'win' : 'live');
  return {value, status, need};
}

// ESPN caches these feeds for ~8s, so polling faster buys nothing.
export const summaryPoll = state => state === 'in' ? 15_000 : state === 'post' ? false : 60_000;
export const scoreboardPoll = games => games.some(g => g.state === 'in') ? 20_000 : games.some(g => g.state === 'pre') ? 60_000 : false;
