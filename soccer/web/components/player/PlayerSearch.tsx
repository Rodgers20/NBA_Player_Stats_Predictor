"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { PlayerSearchHit } from "@/lib/types";

/**
 * Type-ahead player search.
 *
 * Requests are debounced and the previous one is aborted, so fast typing never
 * lands an older response on top of a newer one — the classic search race that
 * shows results for "Yam" after you have finished typing "Yamal".
 */
export function PlayerSearch() {
  const [query, setQuery] = useState("");
  // Results are stored together with the query that produced them, so a stale
  // response can be *derived away* rather than cleared by a setState inside the
  // effect body — which would cascade an extra render on every keystroke.
  const [result, setResult] = useState<{ query: string; hits: PlayerSearchHit[] } | null>(null);
  const [failedQuery, setFailedQuery] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const trimmed = query.trim();
  const active = trimmed.length >= 2;

  useEffect(() => {
    if (!active) return;

    const controller = new AbortController();
    inFlight.current?.abort();
    inFlight.current = controller;

    const timer = setTimeout(async () => {
      try {
        const hits = await api.searchPlayers(trimmed, 20, controller.signal);
        setResult({ query: trimmed, hits });
      } catch (err) {
        if ((err as Error)?.name !== "AbortError") setFailedQuery(trimmed);
      }
    }, 220);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [trimmed, active]);

  // Only show results that belong to what is currently typed.
  const hits = active && result?.query === trimmed ? result.hits : [];
  const state: "idle" | "loading" | "error" =
    failedQuery === trimmed ? "error"
      : active && result?.query !== trimmed ? "loading"
      : "idle";

  const message = useMemo(() => {
    if (state === "error") return "Search is unavailable right now.";
    if (trimmed.length === 0) return "Search any player by name.";
    if (trimmed.length < 2) return "Keep typing — two characters minimum.";
    if (state === "loading") return "Searching…";
    if (!hits.length) return `No player matches “${trimmed}”.`;
    return null;
  }, [state, trimmed, hits.length]);

  return (
    <div>
      <label htmlFor="player-search" className="sr-only">Search players</label>
      <input
        id="player-search"
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search a player — try Yamal, Mbappé, Bellingham"
        autoComplete="off"
        className="w-full rounded-md border border-stroke bg-elevated px-3 py-2.5 text-sm text-ink
                   placeholder:text-ink-muted focus:border-pitch focus:outline-none
                   focus:ring-1 focus:ring-pitch"
      />

      {message ? (
        <p className="px-1 py-3 text-[13px] text-ink-muted" aria-live="polite">{message}</p>
      ) : (
        <ul className="mt-2 divide-y divide-stroke" aria-live="polite">
          {hits.map((h) => (
            <li key={h.player_id}>
              <Link
                href={`/players/${encodeURIComponent(h.player_id)}`}
                className="flex items-center gap-3 px-1 py-2.5 transition-colors hover:bg-elevated"
              >
                <span
                  aria-hidden
                  className="h-9 w-1.5 shrink-0 rounded-full"
                  style={{ background: h.colors.primary }}
                />
                {h.crest ? (
                  <img src={h.crest} alt="" width={22} height={22} className="h-[22px] w-[22px]" />
                ) : (
                  <span className="h-[22px] w-[22px]" />
                )}
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-semibold text-ink">{h.name}</span>
                  <span className="block truncate text-[11px] text-ink-muted">
                    {[h.club, h.position].filter(Boolean).join(" · ")}
                  </span>
                  {h.competitions.length > 0 && (
                    <span className="block truncate text-[10.5px] text-ink-muted opacity-80">
                      {h.competitions.join(" · ")}
                    </span>
                  )}
                </span>
                <span className="shrink-0 text-right">
                  <span className="block text-[13px] font-semibold tabular-nums text-ink">
                    {h.goals ?? 0}
                  </span>
                  <span className="block text-[10px] text-ink-muted">goals</span>
                </span>
                {h.has_full_stats && (
                  <span
                    className="shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold"
                    style={{ borderColor: h.colors.accent, color: h.colors.accent }}
                    title="Full event data: shot map, heatmap and percentile ranks"
                  >
                    Full stats
                  </span>
                )}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
