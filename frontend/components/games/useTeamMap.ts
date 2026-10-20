import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { withPriorSeason } from "@/lib/team-form.mjs";
import type { TeamMap } from "./MatchupCards";

/** Teams keyed by abbreviation. Early in a season it borrows the prior season's games so recent form is never empty. */
export function useTeamMap(league: string): TeamMap {
  const isStatic = process.env.NEXT_PUBLIC_DATA_MODE === "static";
  const teams = useQuery({ queryKey: ["teams", league], queryFn: () => api.teams(league), staleTime: 300_000 });
  const season = Number(teams.data?.season);
  const needsPrior = Number.isFinite(season) && (teams.data?.teams ?? []).some(team => (team.games ?? []).length < 10);
  const prior = useQuery({ queryKey: ["teams", league, season - 1], queryFn: () => api.teams(league, season - 1), staleTime: 300_000, enabled: needsPrior && !isStatic });
  return Object.fromEntries(withPriorSeason(teams.data?.teams ?? [], prior.data?.teams).map(team => [team.abbrev, team]));
}
