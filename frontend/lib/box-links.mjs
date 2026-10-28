const ET = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' });

/** "TEAM|YYYY-MM-DD" (the Eastern date the game log shows) → box-score game id, from the season schedule the team pages use. */
export function boxScoreIndex(teamsResponse) {
  const index = new Map();
  for (const team of teamsResponse?.teams ?? []) {
    for (const game of team.games ?? []) {
      const date = new Date(game.date);
      if (game.id && !Number.isNaN(date.getTime())) index.set(`${team.abbrev}|${ET.format(date)}`, String(game.id));
    }
  }
  return index;
}

/** The box-score page for a game-log row ("BOS vs. BKN" or "BOS @ MIA" on its date), or null when that game isn't in the schedule. */
export function boxScoreHref(index, game, league) {
  const team = String(game?.opponent ?? '').trim().split(/\s+/)[0];
  const id = team && index.get(`${team}|${game.date}`);
  return id ? `/games/box?id=${encodeURIComponent(id)}&league=${league}` : null;
}
