import type { PropsResponse, PlayerChartData } from './types';
import type { ScheduledPlayersResponse } from './api';
export function scheduledPlayersSnapshot(data: ScheduledPlayersResponse, league: string, gameDate: string): ScheduledPlayersResponse;
export function playerKey(name: string): string;
export function filterProps(data: PropsResponse, params?: Record<string, string | number | boolean>): PropsResponse;
export function windowChart(data: PlayerChartData, count: number): PlayerChartData;
export function seriesChart(player: string, series: { games: Record<string, unknown>[] }, stat: string, count?: number): PlayerChartData;
