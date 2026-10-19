// Pure helpers for the research fields on a prop (all inputs may be missing/null).
export function formatOdds(value) {
  if (value == null || !Number.isFinite(value)) return null;
  const n = Math.round(value);
  return n > 0 ? `+${n}` : `${n}`;
}

// 0/1/2/3% ladder -> tier 0..3 (null when edge is unknown).
export function edgeTier(edge) {
  if (edge == null || !Number.isFinite(edge)) return null;
  return edge >= 3 ? 3 : edge >= 2 ? 2 : edge >= 1 ? 1 : 0;
}

// green >= 70, red <= 40, else neutral; null pct -> neutral.
export function hitTone(pct) {
  if (pct == null || !Number.isFinite(pct)) return "neutral";
  return pct >= 70 ? "good" : pct <= 40 ? "bad" : "neutral";
}

const WINDOWS = [["L5", "L5"], ["L10", "L10"], ["L20", "L20"], ["h2h", "H2H"], ["season", "Season"]];
// Chips for windows that have at least one game; [] when hit_windows is absent.
export function hitChips(windows) {
  if (!windows) return [];
  return WINDOWS.flatMap(([key, label]) => {
    const w = windows[key];
    if (!w || !(w.total > 0)) return [];
    const pct = w.pct ?? Math.round((w.hits / w.total) * 100);
    return [{ key, label, hits: w.hits, total: w.total, pct, tone: hitTone(pct) }];
  });
}

export function bestBookKeys(books, side) {
  const field = side === "under" ? "under_price" : "over_price";
  const priced = (books ?? []).filter((b) => b[field] != null);
  if (!priced.length) return new Set();
  const top = Math.max(...priced.map((b) => b[field]));
  return new Set(priced.filter((b) => b[field] === top).map((b) => b.book));
}

export function filterMinStars(props, minStars) {
  return minStars > 0 ? props.filter((p) => (p.stars ?? 0) >= minStars) : props;
}

// Highest edge first; props without an edge sort last. Returns a new array.
export function sortByEdge(props) {
  const key = (p) => (p.edge_pct == null ? -Infinity : p.edge_pct);
  return [...props].sort((a, b) => key(b) - key(a));
}
