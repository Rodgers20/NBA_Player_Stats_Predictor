// ESPN's per-player game log (read live from the browser) turned into plain game rows, plus window averages.

export const GAMELOG_URL = 'https://site.web.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball/athletes';

const made = text => {
  const [m, a] = String(text ?? '').split('-').map(Number);
  return Number.isFinite(m) && Number.isFinite(a) ? [m, a] : [null, null];
};
const num = text => (Number.isFinite(Number(text)) && text !== '' && text != null ? Number(text) : null);

/** Newest-first list of games. Postseason and regular season are merged; a game that appears twice is kept once. */
export function parseGamelog(json) {
  const names = json?.names ?? [];
  const at = key => names.indexOf(key);
  const seen = new Map();
  for (const type of json?.seasonTypes ?? []) {
    for (const cat of type.categories ?? []) {
      for (const ev of cat.events ?? []) {
        const meta = json.events?.[ev.eventId];
        if (!meta || seen.has(ev.eventId)) continue;
        const s = ev.stats ?? [];
        const [fgm, fga] = made(s[at('fieldGoalsMade-fieldGoalsAttempted')]);
        const [tpm, tpa] = made(s[at('threePointFieldGoalsMade-threePointFieldGoalsAttempted')]);
        seen.set(ev.eventId, {
          id: ev.eventId, date: meta.gameDate, home: meta.atVs === 'vs', opp: meta.opponent?.abbreviation ?? meta.opponent?.displayName ?? '',
          oppLogo: meta.opponent?.logo ?? null, result: meta.gameResult ?? null, score: meta.score ?? null,
          min: num(s[at('minutes')]), pts: num(s[at('points')]), reb: num(s[at('totalRebounds')]), ast: num(s[at('assists')]),
          stl: num(s[at('steals')]), blk: num(s[at('blocks')]), tov: num(s[at('turnovers')]), fgm, fga, tpm, tpa,
        });
      }
    }
  }
  return [...seen.values()].sort((a, b) => String(b.date).localeCompare(String(a.date)));
}

/** Mean of each counting stat over the newest `n` games (all games when n is 0). */
export function windowAverages(games, n = 0) {
  const slice = n > 0 ? games.slice(0, n) : games;
  const mean = key => {
    const vals = slice.map(g => g[key]).filter(v => v != null);
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
  };
  return { games: slice.length, min: mean('min'), pts: mean('pts'), reb: mean('reb'), ast: mean('ast'), stl: mean('stl'), blk: mean('blk'), tov: mean('tov'), tpm: mean('tpm') };
}
