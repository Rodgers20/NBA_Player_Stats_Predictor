export function gameState(game) {
  if (['pre', 'in', 'post'].includes(game.status_state)) return game.status_state;
  const status = String(game.status_text ?? '').toLowerCase();
  if (/final|completed/.test(status)) return 'post';
  if (/halftime|quarter|\b[1-4](st|nd|rd|th)\b|\bot\b|in progress/.test(status)) return 'in';
  return 'pre';
}

export function scoreboard(game, prediction) {
  const actual = gameState(game) !== 'pre';
  return {
    away: actual ? game.away_score ?? null : prediction?.predicted_away_score ?? null,
    home: actual ? game.home_score ?? null : prediction?.predicted_home_score ?? null,
    label: actual ? gameState(game) === 'post' ? 'Final score' : 'Live score' : 'Projected score',
  };
}

/** Normalize both quoted moneylines to remove the shared bookmaker margin. */
export function marketProbability(game) {
  const implied = price => typeof price !== 'number' || !Number.isFinite(price) || Math.abs(price) < 100
    ? null : price > 0 ? 100 / (price + 100) : -price / (-price + 100);
  const home = implied(game.home_ml), away = implied(game.away_ml);
  return home == null || away == null ? null : home / (home + away);
}

export function gamePicks(game, prediction) {
  if (!prediction && gameState(game) !== 'pre') return { winner: 'No forecast', spread: '—', total: '—' };
  let spread = game.spread == null ? 'No line' : 'No edge';
  let total = game.total == null ? 'No line' : 'No edge';
  if (prediction?.spread_team && prediction?.spread_pick && game.spread != null) {
    const line = prediction.spread_pick === 'HOME' ? game.spread : -game.spread;
    spread = `${prediction.spread_team} ${line > 0 ? '+' : ''}${line.toFixed(1)}`;
  }
  if (prediction?.total_pick && game.total != null)
    total = `${prediction.total_pick === 'OVER' ? 'Over' : 'Under'} ${game.total.toFixed(1)}`;
  return { winner: prediction?.predicted_winner ?? 'Pending', spread, total };
}

export function shiftDate(day, offset) {
  const date = new Date(`${day}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().slice(0, 10);
}
