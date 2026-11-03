"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ClubDirectoryEntry } from "@/lib/types";

/**
 * Club crest lookup, fetched once per browser session.
 *
 * The UI used to fall back to a club's initials whenever a badge could not be
 * resolved, which produced "BA" for Barça and "C" for Celta. The directory maps
 * every spelling the three sources use onto an official crest URL, so the
 * initials fallback becomes a genuine last resort.
 *
 * One in-flight promise is shared by every component that asks, so a table of
 * twenty rows issues one request rather than twenty.
 */
let pending: Promise<Map<string, string>> | null = null;

/** Aggressively reduce a club name so different sources' spellings collide. */
function key(name: string): string {
  return name
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w && !STOPWORDS.has(w))
    .join(" ");
}

const STOPWORDS = new Set([
  "fc", "cf", "afc", "sc", "ac", "cd", "rc", "rcd", "ud", "sd", "de", "club",
  "calcio", "futbol", "football", "balompie", "deportivo", "real", "sk", "fk",
  "kv", "osc", "ssc", "vfb", "vfl", "pae", "gmbh",
]);

async function load(): Promise<Map<string, string>> {
  const map = new Map<string, string>();
  try {
    const entries: ClubDirectoryEntry[] = await api.clubDirectory();
    for (const e of entries) {
      if (!e.crest) continue;
      // `aliases` carries every spelling the backend's reconciliation reaches,
      // which is what bridges names sharing no tokens with the official one —
      // "Nott'm Forest" and "Nottingham Forest FC", for instance.
      for (const spelling of [e.name, e.short_name, ...(e.aliases ?? [])]) {
        if (!spelling) continue;
        const k = key(spelling);
        if (k && !map.has(k)) map.set(k, e.crest);
      }
    }
  } catch {
    // An unreachable directory just means the initials fallback stays in use.
  }
  return map;
}

export function clubCrests(): Promise<Map<string, string>> {
  pending ??= load();
  return pending;
}

/**
 * The official crest for a club, plus whether the directory is still loading.
 *
 * The loading flag matters: without it a caller renders the legacy crest
 * endpoint on the first paint, that request 404s, and the error handler latches
 * the initials fallback before the real URL ever arrives — which is exactly how
 * Barça kept rendering as "BA" even after the directory existed.
 */
export function useClubCrest(
  name: string | null | undefined,
): { url: string | null; loading: boolean } {
  // Resolution is stored with the name it was made for, so a name change
  // invalidates it by derivation rather than by a setState inside the effect —
  // which would cascade a render on every row of a long table.
  const [resolved, setResolved] = useState<{ name: string; url: string | null } | null>(null);

  useEffect(() => {
    if (!name) return;
    let cancelled = false;
    clubCrests().then((map) => {
      if (cancelled) return;
      const k = key(name);
      // Exact match first, then a containment pass so "Barcelona" finds
      // "FC Barcelona" and "Celta" finds "RC Celta de Vigo".
      let hit = map.get(k);
      if (!hit && k) {
        for (const [candidate, crest] of map) {
          if (candidate === k || candidate.startsWith(`${k} `) || k.startsWith(`${candidate} `)) {
            hit = crest;
            break;
          }
        }
      }
      setResolved({ name, url: hit ?? null });
    });
    return () => { cancelled = true; };
  }, [name]);

  if (!name) return { url: null, loading: false };
  if (resolved?.name !== name) return { url: null, loading: true };
  return { url: resolved.url, loading: false };
}
