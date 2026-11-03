// Server-side reader for the JSON snapshots that replace the FastAPI backend.
// File naming contract: soccer/docs/snapshot-contract.md. Node runtime only (fs).
import { readFile } from "node:fs/promises";
import path from "node:path";
import { snapshotKey } from "./snapshot-key.mjs";

export type Query = Record<string, string | number | undefined | null>;
export interface SnapshotResult { status: number; contentType: string; text: string }

export const snapshotDir = () => process.env.SNAPSHOT_DIR ?? path.join(/*turbopackIgnore: true*/ process.cwd(), "snapshot");

const envelope = (data: unknown, extra: Record<string, unknown> = {}) =>
  JSON.stringify({ success: true, data, error: null, meta: extra });
const failure = (error: string) =>
  JSON.stringify({ success: false, data: null, error, meta: {} });
const JSON_TYPE = "application/json; charset=utf-8";

// ponytail: Map-based FIFO cache of raw text, 64 entries; swap for a real LRU if memory matters.
const CACHE_MAX = 64;
const cache = new Map<string, string | null>();

/** Raw bytes of a file under the snapshot dir, or null when absent / unsafe. */
export async function readSnapshotFile(rel: string): Promise<Buffer | null> {
  if (rel.split("/").some((s) => s === ".." || s === ".")) return null;
  const root = path.resolve(/*turbopackIgnore: true*/ snapshotDir());
  const full = path.resolve(root, rel);
  if (!full.startsWith(root + path.sep)) return null;
  try { return await readFile(/*turbopackIgnore: true*/ full); } catch { return null; }
}

async function readText(rel: string): Promise<string | null> {
  const dir = snapshotDir();
  const hit = cache.get(dir + "|" + rel);
  if (hit !== undefined) return hit;
  const buf = await readSnapshotFile(rel);
  const text = buf ? buf.toString("utf8") : null;
  if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value as string);
  cache.set(dir + "|" + rel, text);
  return text;
}

// ---- player search (mirrors futball/players.search + futball/teams.normalise) ----
const AFFIXES = new Set(("fc cf afc sc ac as ss ssc us usl sv tsv vfl vfb tsg bsc fsv sp cd rc rcd ud sd club " +
  "calcio de futbol football association sport sporting societa spa srl ev real borussia").split(" "));

function normalise(name: string): string {
  if (!name) return "";
  const s = name.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase()
    .replace(/&/g, " and ").replace(/[^a-z0-9\s]/g, " ");
  const tokens = s.split(/\s+/).filter(Boolean);
  const kept = tokens.filter((t) => !AFFIXES.has(t));
  return (kept.length ? kept : tokens).join(" ");
}

interface Hit { name: string; has_full_stats?: boolean; goals?: number | null }

async function searchPlayers(q: string, limit: number): Promise<string> {
  const needle = normalise(q.trim());
  if (q.trim().length < 2 || !needle) return envelope([]);
  const raw = await readText("players/search-index.json");
  if (!raw) return failure("not in snapshot: players/search-index.json");
  const hits = (JSON.parse(raw).data as Hit[]).filter((h) => normalise(h.name).includes(needle));
  hits.sort((a, b) =>
    Number(!a.has_full_stats) - Number(!b.has_full_stats) ||
    (b.goals ?? 0) - (a.goals ?? 0) ||
    (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return envelope(hits.slice(0, limit));
}

/** Resolve one API GET against the snapshot. Never throws. */
export async function snapshotGet(apiPath: string, query: Query = {}): Promise<SnapshotResult> {
  const p = "/" + apiPath.replace(/^\/+|\/+$/g, "");
  const ok = (text: string): SnapshotResult => ({ status: 200, contentType: JSON_TYPE, text });

  if (p === "/live/now") return ok(envelope([]));
  if (p === "/live/counts") return ok(envelope({ total: 0, by_competition: {}, matches: [] }));
  if (p === "/live/scores") {
    // Snapshot build: no live source. stale=true makes the scoreboard say availability is unconfirmed.
    return ok(envelope({ source: "snapshot", matches: [], competitions: [], stale: true, poll_seconds: 45 }));
  }
  if (p === "/live" || p.startsWith("/live/")) return { status: 501, contentType: JSON_TYPE, text: failure("live replay is not available in the snapshot build") };
  if (p === "/players/search") {
    const limit = Math.min(Math.max(Number(query.limit) || 20, 1), 50);
    return ok(await searchPlayers(String(query.q ?? ""), limit));
  }

  const text = await readText(snapshotKey(p, query));
  if (text === null) return { status: 404, contentType: JSON_TYPE, text: failure(`not in snapshot: ${p}`) };
  return {
    status: 200,
    contentType: p.endsWith(".ics") ? "text/calendar; charset=utf-8" : JSON_TYPE,
    text,
  };
}

// ---- images ----
const MIME: Record<string, string> = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", svg: "image/svg+xml", webp: "image/webp",
};

export async function snapshotImage(kind: "crest" | "headshot", name: string) {
  const raw = await readText("img/manifest.json");
  if (!raw) return null;
  const map = (JSON.parse(raw) as Record<string, Record<string, string>>)[kind] ?? {};
  let rel = map[name];
  if (!rel) {
    const lower = name.toLowerCase();
    const k = Object.keys(map).find((x) => x.toLowerCase() === lower);
    rel = k ? map[k] : "";
  }
  if (!rel) return null;
  const buf = await readSnapshotFile("img/" + rel);
  if (!buf) return null;
  return { buf, type: MIME[rel.split(".").pop()!.toLowerCase()] ?? "application/octet-stream" };
}
