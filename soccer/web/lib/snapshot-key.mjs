// Snapshot filename for an API request. Contract: soccer/docs/snapshot-contract.md
// (the Python exporter implements the same function).

const KEEP = /[A-Za-z0-9._-]/;

/** UTF-8 bytes; anything outside [A-Za-z0-9._-] becomes ~XX (uppercase hex). */
export function enc(s) {
  let out = "";
  for (const b of new TextEncoder().encode(String(s))) {
    const ch = String.fromCharCode(b);
    out += b < 128 && KEEP.test(ch) ? ch : "~" + b.toString(16).toUpperCase().padStart(2, "0");
  }
  return out;
}

/** @param {string} path  @param {Record<string, unknown>} [query] */
export function snapshotKey(path, query = {}) {
  const segs = path.replace(/^\/+|\/+$/g, "").split("/").map(enc).join("/");
  const pairs = Object.keys(query)
    .filter((k) => query[k] !== undefined && query[k] !== null && query[k] !== "")
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
    .map((k) => `${enc(k)}=${enc(query[k])}`);
  const ext = path.replace(/\/+$/, "").endsWith(".ics") ? "" : ".json";
  return segs + (pairs.length ? "@" + pairs.join("&") : "") + ext;
}
