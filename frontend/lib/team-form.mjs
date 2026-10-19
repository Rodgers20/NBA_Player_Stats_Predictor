/** Normalize provider order before selecting the latest completed scoring games. */
export function recentScoringGames(games, limit = 20) {
  return games.filter(game => game.points_for != null && game.points_against != null)
    .sort((a, b) => Date.parse(a.date) - Date.parse(b.date))
    .slice(-limit);
}
