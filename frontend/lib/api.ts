import type { TeamsResponse } from './team-types';
import type { StandingsResponse } from './standings-types';
import type { BoxScore } from './boxscore-types';
import type { PropsResponse, GamesResponse, PlayerChartData, PlayerStats, GamePredictionsResponse, HighlightsResponse, ScheduleIndex, PerformanceResponse, LineHistory, DefenseRankings, TrendsResponse, MatchupResponse } from "./types";
import { playerKey, filterProps, seriesChart, scheduledPlayersSnapshot, filterDefense, filterTrends } from './snapshot.mjs';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, '');
const STATIC = process.env.NEXT_PUBLIC_DATA_MODE === 'static';
type Params = Record<string, string | number | boolean>;

async function get<T>(path: string, params?: Params): Promise<T> {
  const query = new URLSearchParams(Object.entries(params ?? {}).filter(([,v]) => v !== '').map(([k,v]) => [k,String(v)]));
  const res = await fetch(`${STATIC ? '' : BASE}${path}${query.size ? '?' + query : ''}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Data is unavailable (${res.status}). Please try again later.`);
  return res.json() as Promise<T>;
}
const snapshot = <T,>(league: string, file: string) => get<T>(`/data/${league}/${file}.json`);

export const api = {
  scheduledPlayers: async (league: string, gameDate: string): Promise<ScheduledPlayersResponse> => {
    if (!STATIC) return get<ScheduledPlayersResponse>('/api/players/scheduled', { league, game_date: gameDate });
    try {
      return scheduledPlayersSnapshot(await snapshot<ScheduledPlayersResponse>(league, 'scheduled-players'), league, gameDate);
    } catch {
      return { league, game_date: gameDate, players: [], message: 'Player suggestions are unavailable in this snapshot.' };
    }
  },
  props: async (params: Params = {}) => STATIC
    ? filterProps(await snapshot<PropsResponse>(String(params.league ?? 'nba'), 'props'), params)
    : get<PropsResponse>('/api/props', params),
  // Line history is a live-only research aid; static snapshots deliberately carry none.
  lineHistory: async (player: string, stat: string, league = 'nba'): Promise<LineHistory> => STATIC
    ? { league, player, stat, snapshots: [] }
    : get<LineHistory>('/api/props/line-history', { player, stat, league }),
  defense: async (league = 'nba', stat = ''): Promise<DefenseRankings> => {
    if (!STATIC) return get<DefenseRankings>('/api/defense', { league, stat });
    try {
      return filterDefense(await snapshot<DefenseRankings>(league, 'defense'), stat);
    } catch {
      return { league, stat: stat || null, season: null, rankings: [] };
    }
  },
  trends: async (league = 'nba', params: Params = {}): Promise<TrendsResponse> => {
    if (!STATIC) return get<TrendsResponse>('/api/trends', { league, ...params });
    try {
      return filterTrends(await snapshot<TrendsResponse>(league, 'trends'), params);
    } catch {
      return { league, count: 0, trends: [] };
    }
  },
  altLines: (league = 'nba') => STATIC ? snapshot<AltLinesResponse>(league, 'alt-lines') : get<AltLinesResponse>('/api/props/alt-lines', { league }),
  propsRecord: () => get<PropsRecordResponse>(STATIC ? '/data/props-record.json' : '/api/props/record'),
  predictions: (league = 'nba', gameDate?: string) => STATIC ? snapshot<GamePredictionsResponse>(league, gameDate ? `schedule/${gameDate}-predictions` : 'predictions') : get<GamePredictionsResponse>('/api/games/predictions', { league, ...(gameDate ? {game_date: gameDate} : {}) }),
  budget: (league = 'nba') => get<{ configured: boolean; daily: number; daily_limit: number; monthly: number; monthly_limit: number; remaining: number | null; message: string }>('/api/props/budget', { league }),
  refreshProps: async (league = 'nba') => {
    if (STATIC) throw new Error('Refreshing quotes requires the connected app.');
    const response = await fetch(`${BASE}/api/props/refresh?league=${league}&fetch_odds=true`, { method: 'POST' });
    if (!response.ok) throw new Error('Could not refresh quotes. Please try again later.');
    return response.json() as Promise<{ count: number; message: string; skipped: { reason: string }[]; budget?: { message: string } }>;
  },
  games: (league = 'nba', gameDate?: string) => STATIC ? snapshot<GamesResponse>(league, gameDate ? `schedule/${gameDate}` : 'games') : get<GamesResponse>('/api/games', { league, ...(gameDate ? {game_date: gameDate} : {}) }),
  matchup: (league: string, home: string, away: string) => STATIC
    ? snapshot<MatchupResponse>(league, `matchup/${away}-${home}`)
    : get<MatchupResponse>('/api/games/matchup', { league, home, away }),
  teams: (league = 'nba', season?: number) => STATIC ? snapshot<TeamsResponse>(league, season ? `teams-${season}` : 'teams') : get<TeamsResponse>('/api/teams', {league, ...(season ? {season} : {})}),
  performance: (league = 'nba') => STATIC ? snapshot<PerformanceResponse>(league, 'performance') : get<PerformanceResponse>('/api/performance', {league}),
  standings: (league = 'nba') => STATIC ? snapshot<StandingsResponse>(league, 'standings') : get<StandingsResponse>('/api/standings', {league}),
  boxscore: (league: string, gameId: string) => STATIC ? snapshot<BoxScore>(league, `boxscore/${gameId}`) : get<BoxScore>(`/api/games/${encodeURIComponent(gameId)}/boxscore`, {league}),
  highlights: (league = 'nba') => STATIC ? snapshot<HighlightsResponse>(league, 'highlights') : get<HighlightsResponse>('/api/highlights', {league}),
  scheduleIndex: (league = 'nba') => STATIC ? snapshot<ScheduleIndex>(league, 'schedule-index') : Promise.resolve(null),
  refreshGameLines: async (league: 'nba' | 'wnba') => {
    if (STATIC) throw new Error('Refreshing game lines requires the connected app.');
    const response = await fetch(`${BASE}/api/games/refresh-lines?league=${encodeURIComponent(league)}`, { method: 'POST' });
    if (!response.ok) throw new Error('Could not refresh game lines.');
    return response.json() as Promise<{ count: number; status: string; message: string }>;
  },
  hitrates: () => STATIC ? snapshot<HitRatesResponse>('wnba', 'hitrates') : get<HitRatesResponse>('/api/wnba/hitrates'),
  playerChart: async (player: string, stat = 'PTS', games = 20, league = 'nba') => STATIC
    ? seriesChart(player, (await snapshot<{ series: { games: Record<string, unknown>[] } }>(league, `player/${playerKey(player)}`)).series, stat, games)
    : get<PlayerChartData>(`/api/player/${encodeURIComponent(player)}/chart-data`, { stat, games, league }),
  playerStats: async (player: string, league = 'nba') => STATIC
    ? (await snapshot<{ stats: PlayerStats }>(league, `player/${playerKey(player)}`)).stats
    : get<PlayerStats>(`/api/player/${encodeURIComponent(player)}/stats`, { league }),
  players: async (q = '', league = 'nba') => {
    if (!STATIC) return get<{players: string[]; headshots?: Record<string, string>}>('/api/players', { q, league });
    const data = await snapshot<{players: string[]; headshots?: Record<string, string>}>(league, 'players');
    return { players: data.players.filter(name => name.toLowerCase().includes(q.toLowerCase())), headshots: data.headshots ?? {} };
  },
};

export interface ScheduledPlayersResponse {
  league: string;
  game_date: string;
  players: { name: string; team: string; opponent?: string }[];
  message?: string;
}

export interface HitRatesResponse {
  target_date: string;
  message: string | null;
  games: { matchup: string; home: string; away: string; total_count: number; entries: {
    player_name: string; team: string; stat: string; threshold: number; hits: number; games: number;
  }[] }[];
}

export interface AltLinesResponse {
  alt_lines: { team: string; player: string; stat: string; stat_label: string; threshold: number; trend: string }[];
  count: number;
  target_date: string | null;
}

export interface PropsRecordResponse {
  hit: number;
  miss: number;
  total: number;
  pct: number;
  recent_7d: number;
  by_stat: Record<string, { hit: number; total: number; pct: number }>;
}
