"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { LiveMatch } from "@/lib/types";

/**
 * The running score for a club currently playing, in that club's own terms.
 *
 * The score reads **from this row's point of view** — the row team's goals
 * first — rather than as the match's home–away scoreline. The reference design
 * shows the raw scoreline in both rows, so a side losing 1-2 displays "2 - 1"
 * in red, which reads as winning and is contradicted only by the colour. Their
 * own goals first means the number and the colour say the same thing.
 *
 * Green ahead, red behind, grey level.
 */
const TONE = {
  winning: { bg: "var(--color-form-w, #09AF51)", fg: "var(--color-base)" },
  losing: { bg: "var(--color-form-l, #FF3B47)", fg: "var(--color-base)" },
  drawing: { bg: "var(--color-form-d, #7A8290)", fg: "var(--color-base)" },
} as const;

export type LiveState = keyof typeof TONE;

export function liveStateFor(team: string, match: LiveMatch): LiveState | null {
  const { home_goals: hg, away_goals: ag } = match;
  if (hg === null || hg === undefined || ag === null || ag === undefined) return null;
  const atHome = match.home_team === team;
  const [own, other] = atHome ? [hg, ag] : [ag, hg];
  if (own > other) return "winning";
  if (own < other) return "losing";
  return "drawing";
}

/** Finds the in-play match a club is involved in, if any. */
export function liveMatchFor(team: string, live: LiveMatch[] | undefined): LiveMatch | undefined {
  return live?.find((m) => m.home_team === team || m.away_team === team);
}

export function LiveScoreChip({ team, match }: { team: string; match: LiveMatch }) {
  const state = liveStateFor(team, match);
  if (!state) return null;

  const atHome = match.home_team === team;
  const own = atHome ? match.home_goals! : match.away_goals!;
  const other = atHome ? match.away_goals! : match.home_goals!;
  const opponent = atHome ? match.away_team : match.home_team;
  const tone = TONE[state];

  return (
    <span
      className="inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 font-mono text-[11px] font-semibold tabular-nums"
      style={{ background: tone.bg, color: tone.fg }}
      title={
        `Live: ${own} - ${other} ${atHome ? "v" : "at"} ${opponent}` +
        (match.minutes_elapsed !== null ? ` · ${match.minutes_elapsed}'` : "")
      }
    >
      <span className="live-dot !h-1.5 !w-1.5" aria-hidden />
      {own} - {other}
      <span className="sr-only">
        live, {state === "winning" ? "ahead" : state === "losing" ? "behind" : "level"}
      </span>
    </span>
  );
}


/**
 * A table row's live chip, kept current without reloading the page.
 *
 * The server-rendered score seeds the query, so the first paint is correct and
 * there is no flash of nothing. Polling is every 45 seconds to match the
 * server's own live cache TTL — faster would only re-read the same cached
 * payload. It is affordable at all because football-data.org has no daily
 * request cap; on API-Football's 100-a-day tier this feature could not exist.
 */
const POLL_MS = 45_000;

export function LiveScoreCell({
  team, live,
}: {
  team: string;
  /** What the server saw at request time; seeds the first paint. */
  live: LiveMatch[] | undefined;
}) {
  const { data } = useQuery({
    queryKey: ["live-counts"],
    queryFn: () => api.liveCounts(),
    refetchInterval: POLL_MS,
    // Every chip on the table shares one query, so twenty rows cost one
    // request rather than twenty.
    staleTime: POLL_MS,
  });

  const match = liveMatchFor(team, data ? data.matches : live);
  if (!match) return null;
  return <LiveScoreChip team={team} match={match} />;
}
