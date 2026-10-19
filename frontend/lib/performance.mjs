/** Missing evidence stays visibly missing, never zero or a claimed hit rate. */
export const metric = (value, digits = 1) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
export const percentage = (value, digits = 1) => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—';
export const seasonLabel = (year, league) => league === 'nba' ? `${year}–${String(year + 1).slice(-2)}` : String(year);
export function calibrationPoints(bins = []) {
  return bins.filter(bin => bin.count > 0 && Number.isFinite(bin.predicted) && Number.isFinite(bin.actual) && bin.predicted >= 0 && bin.predicted <= 1 && bin.actual >= 0 && bin.actual <= 1).sort((a, b) => a.predicted - b.predicted);
}
export function selectedSeason(live, requested) {
  return live.seasons.find(item => item.season === requested) ?? live.seasons.find(item => item.season === live.current_season) ?? live.seasons[0];
}
