import type { TeamGame, BasketballTeam } from "./team-types";
export function recentScoringGames(games: TeamGame[], limit?: number): TeamGame[];
export function withPriorSeason(teams: BasketballTeam[], priorTeams: BasketballTeam[] | undefined, minGames?: number): BasketballTeam[];
