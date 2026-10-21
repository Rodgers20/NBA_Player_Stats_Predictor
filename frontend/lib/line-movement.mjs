/** Pure helpers for the line-movement tab. Snapshots are ascending by `ts` (seconds). */
export function bookSeries(snapshots) {
  const series = new Map();
  for (const snap of snapshots) series.set(snap.book, [...(series.get(snap.book) ?? []), snap]);
  return series;
}

/** The book with the most snapshots; ties go to the one seen first. */
export function primaryBook(snapshots) {
  let best = null;
  for (const [book, points] of bookSeries(snapshots)) if (!best || points.length > best.count) best = { book, count: points.length };
  return best?.book ?? null;
}

export function movement(points) {
  if (!points.length) return null;
  const first = points[0], last = points[points.length - 1];
  return { opening: first.line, current: last.line, change: Math.round((last.line - first.line) * 100) / 100, samples: points.length };
}
