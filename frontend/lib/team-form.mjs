/**
 * Early in a season a team has few results. Prepend the prior season's games so "last N" spans the
 * boundary like bettors expect. Teams with enough games, or without a prior-season match, are untouched.
 */
export function withPriorSeason(teams, priorTeams, minGames = 10) {
  if (!priorTeams) return teams;
  const prior = new Map(priorTeams.map(team => [team.abbrev, team.games ?? []]));
  return teams.map(team => {
    const have = (team.games ?? []).filter(game => game.points_for != null && game.points_against != null).length;
    const earlier = prior.get(team.abbrev);
    return have >= minGames || !earlier ? team : { ...team, games: [...earlier, ...(team.games ?? [])] };
  });
}

/** Normalize provider order before selecting the latest completed scoring games. */
export function recentScoringGames(games, limit = 20) {
  return games.filter(game => game.points_for != null && game.points_against != null)
    .sort((a, b) => Date.parse(a.date) - Date.parse(b.date))
    .slice(-limit);
}
