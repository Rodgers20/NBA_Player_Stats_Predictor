const STATS = new Set(['PTS', 'AST', 'REB', 'FG3M', 'BLK', 'STL', 'PTS+AST', 'PTS+REB', 'REB+AST', 'PTS+REB+AST', 'STL+BLK',
  'PTS_1Q', 'REB_1Q', 'AST_1Q', 'FG3M_1Q']);
const NAMED = new Set(['h2h', 'home', 'away', 'current', 'previous']);

/** Stat, window and line from a URL query. Anything unrecognised keeps the default. */
export function parseAnalysisView(search, defaults) {
  const params = new URLSearchParams(search);
  const rawStat = params.get('stat')?.replace(/ /g, '+'); // a typed "+" arrives as a space
  const rawPeriod = params.get('period');
  const rawLine = params.get('line');
  const line = rawLine !== null && rawLine.trim() !== '' ? Number(rawLine) : NaN;
  return {
    stat: rawStat && STATS.has(rawStat) ? rawStat : defaults.stat,
    period: rawPeriod !== null && /^(5|10|20)$/.test(rawPeriod) ? Number(rawPeriod) : rawPeriod !== null && NAMED.has(rawPeriod) ? rawPeriod : defaults.period,
    line: Number.isFinite(line) && line >= 0 && line <= 200 ? Math.round(line * 2) / 2 : defaults.line,
  };
}

/** Only the parts that differ from the defaults, in a stable order, without a leading "?". */
export function analysisViewToSearch(view, defaults) {
  const params = new URLSearchParams();
  if (view.stat !== defaults.stat) params.set('stat', view.stat);
  if (view.period !== defaults.period) params.set('period', String(view.period));
  if (view.line != null && view.line !== defaults.line) params.set('line', String(view.line));
  return params.toString();
}
