"use client";

import { useQueries } from '@tanstack/react-query';
import { api } from '@/lib/api';

/** Player name → self-hosted photo for each league on the slip, on the same cache key the Analysis page uses. */
export function useHeadshots(leagues: string[]): Record<string, Record<string, string>> {
  const unique = [...new Set(leagues)];
  const results = useQueries({ queries: unique.map(league => ({
    queryKey: ['players', 'all', league], queryFn: () => api.players('', league), staleTime: 3_600_000,
  })) });
  return Object.fromEntries(unique.map((league, i) => [league, results[i]?.data?.headshots ?? {}]));
}
