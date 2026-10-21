import { recentScoringGames } from './team-form.mjs';

/**
 * Each team's recent results framed against today's lines, so a ribbon can show
 * "would this team have cleared today's number in its last N games".
 * Spread is the home handicap (home +1.5 => spread 1.5), so a team covers when margin + handicap > 0,
 * i.e. margin "Over" the negated handicap. Missing lines give null, never a zero line.
 */
export function marketForm(game, homeGames, awayGames, limit = 10) {
  const home = recentScoringGames(homeGames, limit);
  const away = recentScoringGames(awayGames, limit);
  const margin = item => item.points_for - item.points_against;
  const points = item => item.points_for + item.points_against;
  return {
    moneyline: {
      home: { values: home.map(margin), line: 0 },
      away: { values: away.map(margin), line: 0 },
    },
    spread: game.spread == null ? null : {
      home: { values: home.map(margin), line: -game.spread },
      away: { values: away.map(margin), line: game.spread },
    },
    total: game.total == null ? null : { home: home.map(points), away: away.map(points), line: game.total },
  };
}
