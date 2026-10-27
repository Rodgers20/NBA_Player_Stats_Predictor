"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { boxScoreHref, boxScoreIndex } from "@/lib/box-links.mjs";

/** game-log row → its box-score page, from the season schedule (same cache key as the Games and Teams pages). */
export function useBoxLinks(league: string) {
  const teams = useQuery({ queryKey: ["teams", league], queryFn: () => api.teams(league), staleTime: 300_000 });
  const index = useMemo(() => boxScoreIndex(teams.data), [teams.data]);
  return (game: { date: string; opponent: string }) => boxScoreHref(index, game, league);
}
