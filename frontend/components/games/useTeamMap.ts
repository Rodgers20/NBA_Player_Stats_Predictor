import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { withPriorSeason } from "@/lib/team-form.mjs";
import type { TeamMap } from "./MatchupCards";

/** Teams keyed by abbreviation, plus whether loading has settled. Early in a season it borrows the prior season's games so recent form is never empty. */
export function useTeamMap(league: string): { teamMap: TeamMap; ready: boolean } {
  const teams = useQuery({ queryKey: ["teams", league], queryFn: () => api.teams(league), staleTime: 300_000 });
  const season = Number(teams.data?.season);
  const needsPrior = Number.isFinite(season) && (teams.data?.teams ?? []).some(team => (team.games ?? []).length < 10);
  const prior = useQuery({ queryKey: ["teams", league, season - 1], queryFn: () => api.teams(league, season - 1), staleTime: 300_000, enabled: needsPrior });
  // Ready once team data (and prior-season data, when needed) has settled, success or failure, so cards never reflow when it lands.
  const ready = !teams.isPending && (!needsPrior || !prior.isPending);
  const teamMap: TeamMap = Object.fromEntries(withPriorSeason(teams.data?.teams ?? [], prior.data?.teams).map(team => [team.abbrev, team]));
  return { teamMap, ready };
}
