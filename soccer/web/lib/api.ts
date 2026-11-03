import type {
  BacktestRow, FormMatch, H2HMatch, Fixture, League, LeagueMatches, LeagueSeason, LeagueStandings,
  LiveCounts, LiveMatch, MatchdayBoard, PlayerLeaders,
  PredictCompetition, PredictMetrics, Prediction,
  Competition, Envelope, FreezeFrame, Heatmap, Match, MatchEvent,
  MomentumPoint, Pass, PassNetwork, Shot, StandingRow, StatRow, TopPlayer,
  ScorerRow, AssistRow, ContributionRow,
  PlayerSearchHit,
  PlayerProfile,
  ClubDirectoryEntry,
  SeasonHonour,
  MetricSpec,
  LeaderBoard,
  CompetitionNews,
  TeamOfTheWeekData,
  TeamProfile,
} from "./types";

/** "/soccer" in the deployed build, "" for plain localhost dev. */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

// Browser: the snapshot API route handlers (app/api). Server: read the snapshot directly.
export const API_BASE = `${BASE_PATH}/api`;

const ASSET_BASE = API_BASE;

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly path: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  let status: number;
  let text: string;
  if (typeof window === "undefined") {
    const { snapshotGet } = await import("./snapshot");
    const u = new URL(path, "http://snapshot.local");
    const r = await snapshotGet(u.pathname, Object.fromEntries(u.searchParams));
    status = r.status;
    text = r.text;
  } else {
    let res: Response;
    try {
      res = await fetch(`${API_BASE}${path}`, { signal });
    } catch {
      throw new ApiError("Cannot load the football data snapshot.", 0, path);
    }
    status = res.status;
    text = await res.text();
  }
  let body: Envelope<T>;
  try {
    body = JSON.parse(text) as Envelope<T>;
  } catch {
    throw new ApiError(`Malformed response from ${path}`, status, path);
  }
  if (status >= 400 || !body.success) {
    throw new ApiError(body.error ?? `Request failed (${status})`, status, path);
  }
  return body.data as T;
}

const qs = (params: Record<string, string | number | undefined | null>) => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : "";
};

export const api = {
  latestLineup: (team: string, competition_code: string) => get<import("./types").SeasonXi>(`/lineups/latest${qs({ team, competition_code })}`),
  health: () => get<{ status: string; competitions: number; data_root: string }>("/health"),
  quality: () =>
    get<{ name: string; value: number; passed: boolean; detail: string }[]>("/quality"),
  competitions: () => get<Competition[]>("/competitions"),
  matches: (p: { competition_id?: number; season_id?: number; limit?: number } = {}) =>
    get<Match[]>(`/matches${qs(p)}`),
  match: (id: number) => get<Match>(`/matches/${id}`),
  stats: (id: number) => get<StatRow[]>(`/matches/${id}/stats`),
  events: (id: number) => get<MatchEvent[]>(`/matches/${id}/events`),
  momentum: (id: number, window = 5) =>
    get<MomentumPoint[]>(`/matches/${id}/momentum${qs({ window })}`),
  passes: (id: number, p: { team?: string; limit?: number } = {}) =>
    get<Pass[]>(`/matches/${id}/passes${qs(p)}`),
  passNetwork: (id: number, team: string) =>
    get<PassNetwork>(`/matches/${id}/pass-network${qs({ team })}`),
  shots: (id: number) => get<Shot[]>(`/matches/${id}/shots`),
  heatmap: (id: number, p: { team?: string; bins_x?: number; bins_y?: number } = {}) =>
    get<Heatmap>(`/matches/${id}/heatmap${qs(p)}`),
  freezeFrames: (id: number, limit = 400) =>
    get<FreezeFrame[]>(`/matches/${id}/freeze-frames${qs({ limit })}`),
  lineupPositions: (id: number) =>
    get<{ team: string; player: string; position: string | null; x: number; y: number;
          touches: number; first_minute: number; last_minute: number }[]>(
      `/matches/${id}/lineup-positions`),
  topPlayers: (id: number, limit = 5) =>
    get<TopPlayer[]>(`/matches/${id}/top-players${qs({ limit })}`),
  standings: (p: { competition_id?: number; stage?: string } = {}) =>
    get<StandingRow[]>(`/standings${qs(p)}`),
  topScorers: (p: { competition_id?: number; limit?: number } = {}) =>
    get<ScorerRow[]>(`/players/top-scorers${qs(p)}`),
  topAssists: (p: { competition_id?: number; limit?: number } = {}) =>
    get<AssistRow[]>(`/players/top-assists${qs(p)}`),
  topContributions: (p: { competition_id?: number; limit?: number } = {}) =>
    get<ContributionRow[]>(`/players/top-contributions${qs(p)}`),
  // ---- league (club competition) data ----
  leagues: () => get<League[]>("/leagues"),
  leagueSeasons: (code: string) => get<LeagueSeason[]>(`/leagues/${code}/seasons`),
  leagueStandings: (code: string, season_year?: number) =>
    get<LeagueStandings>(`/leagues/${code}/standings${qs({ season_year })}`),
  leagueMatches: (code: string, p: { season_year?: number; limit?: number } = {}) =>
    get<LeagueMatches>(`/leagues/${code}/matches${qs(p)}`),
  leagueFixtures: (code: string, days = 10) =>
    get<Fixture[]>(`/leagues/${code}/fixtures${qs({ days })}`),
  fixtures: (p: { competition_code?: string; days?: number } = {}) =>
    get<Fixture[]>(`/fixtures${qs(p)}`),
  liveNow: (competition_code?: string) =>
    get<LiveMatch[]>(`/live/now${qs({ competition_code })}`),
  liveCounts: () => get<LiveCounts>("/live/counts"),
  headToHead: (code: string, home: string, away: string, limit = 6) =>
    get<H2HMatch[]>(`/leagues/${code}/head-to-head${qs({ home, away, limit })}`),
  teamForm: (code: string, team: string, limit = 6) =>
    get<FormMatch[]>(`/leagues/${code}/team-form${qs({ team, limit })}`),
  playerLeaders: (competition_code: string, limit = 10, season_year?: number) =>
    get<PlayerLeaders>(`/players/leaders${qs({ competition_code, limit, season_year })}`),
  searchPlayers: (q: string, limit = 20, signal?: AbortSignal) =>
    get<PlayerSearchHit[]>(`/players/search${qs({ q, limit })}`, signal),
  playerDetails: (playerId: string) =>
    get<import("./types").PlayerDetails>(`/players/${encodeURIComponent(playerId)}/details`),
  playerProfile: (playerId: string) =>
    get<PlayerProfile>(`/players/profile/${encodeURIComponent(playerId)}`),
  competitionNews: (code: string, limit = 12) =>
    get<CompetitionNews>(`/competitions/${code}/news${qs({ limit })}`),
  teamOfTheWeek: (code: string, match_week?: number, season_year?: number) =>
    get<TeamOfTheWeekData>(
      `/competitions/${code}/team-of-the-week${qs({ match_week, season_year })}`),
  ratingLeaders: (limit = 5) =>
    get<{ player_id: number; player: string; team: string | null; rating: number }[]>(
      `/ratings/leaders${qs({ limit })}`),
  calendarUrl: (code: string) => `${ASSET_BASE}/leagues/${code}/calendar.ics`,
  teamMetrics: (code: string) =>
    get<MetricSpec[]>(`/competitions/${code}/team-metrics`),
  teamLeaders: (code: string, metric: string, limit = 3) =>
    get<LeaderBoard>(`/competitions/${code}/team-leaders${qs({ metric, limit })}`),
  playerMetrics: (code: string) =>
    get<MetricSpec[]>(`/competitions/${code}/player-metrics`),
  playerLeaderBoard: (code: string, metric: string, limit = 3) =>
    get<LeaderBoard>(`/competitions/${code}/player-leaders${qs({ metric, limit })}`),
  honours: (code: string) => get<SeasonHonour[]>(`/leagues/${code}/honours`),
  teamProfile: (team: string, competition?: string) =>
    get<TeamProfile>(`/teams/profile${qs({ team, competition })}`),
  teamCalendarUrl: (team: string) =>
    `${ASSET_BASE}/teams/calendar.ics?team=${encodeURIComponent(team)}`,
  clubDirectory: () => get<ClubDirectoryEntry[]>("/teams/directory"),
  cleanSheets: (code: string, limit = 20) =>
    get<{ rows: { team: string; clean_sheets: number; played: number;
      goals_conceded: number; position: number }[]; season: string | null;
      note: string | null }>(`/leagues/${code}/clean-sheets${qs({ limit })}`),
  headshotUrl: (name: string) =>
    `${ASSET_BASE}/players/headshot?name=${encodeURIComponent(name)}&v=api-football-1`,
  crestUrl: (name: string) =>
    `${ASSET_BASE}/teams/crest?name=${encodeURIComponent(name)}`,
  predictFixtureList: (competition_code: string, days = 21) =>
    get<Fixture[]>(`/predict/fixture-list${qs({ competition_code, days })}`),
  predictByFixture: (fixture_key: string) =>
    get<Prediction & { fixture?: { date: string; time: string | null;
      kickoff_utc: string | null; status: string; referee: string | null;
      odds: { home: number | null; draw: number | null; away: number | null } } }>(
      `/predict/by-fixture${qs({ fixture_key })}`),
  predictMatchday: (competition_code: string, matchday?: number) =>
    get<MatchdayBoard>(`/predict/matchday${qs({ competition_code, matchday })}`),
  predictCompetitions: () => get<PredictCompetition[]>("/predict/competitions"),
  predictTeams: (competition_code: string) =>
    get<string[]>(`/predict/teams${qs({ competition_code })}`),
  predictFixture: (competition_code: string, home: string, away: string) =>
    get<Prediction>(`/predict/fixture${qs({ competition_code, home, away })}`),
  predictMetrics: (competition_code?: string) => get<PredictMetrics>(`/predict/metrics${qs({ competition_code })}`),
  predictBacktest: (competition_code: string, limit = 10) =>
    get<BacktestRow[]>(`/predict/backtest${qs({ competition_code, limit })}`),
  // Live replay needs the streaming backend, which the snapshot build does not have.
  startLive: async (...args: [id: number, speed?: number]): Promise<never> => {
    void args;
    throw new ApiError("Live replay is not available in the snapshot build.", 501, "/live");
  },
  stopLive: async (...args: [id: number]) => (void args, { success: true }),
};

export const wsUrl = (matchId: number, since = 0) =>
  `${API_BASE}/ws/match/${matchId}?since=${since}`;
