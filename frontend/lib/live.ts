import { useQuery } from "@tanstack/react-query";
import { ESPN, parseScoreboard, parseSummary, scoreboardPoll, summaryPoll, type LiveBox, type LiveGame } from "./espn-live.mjs";

/**
 * Live ESPN feeds polled from the visitor's browser, so every visitor uses their own quota.
 * Each feed is one shared query: the box score, play feed and tracked bets reuse the same request.
 * React Query pauses these intervals in hidden tabs; errors back off to a minute.
 */
async function espn(league: string, path: string, params: Record<string, string>, signal: AbortSignal) {
  const response = await fetch(`${ESPN}/${league}/${path}?${new URLSearchParams(params)}`, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(12_000)]),
  });
  if (!response.ok) throw new Error(`ESPN is unavailable (${response.status}).`);
  return response.json();
}

const easternDate = (offsetDays = 0) => new Date(Date.now() + offsetDays * 86_400_000).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
export const easternToday = () => easternDate();
export const easternYesterday = () => easternDate(-1);
const BACKOFF = 60_000;

export function summaryQuery(league: string, eventId: string, enabled = true) {
  return {
    queryKey: ["espn-summary", league, eventId],
    queryFn: async ({ signal }: { signal: AbortSignal }): Promise<LiveBox> => parseSummary(league, await espn(league, "summary", { event: eventId }, signal)),
    enabled: enabled && /^\d{6,12}$/.test(eventId),
    refetchInterval: ({ state }: { state: { error: unknown; data?: LiveBox } }) => state.error ? BACKOFF : summaryPoll(state.data?.state),
    staleTime: 10_000,
    retry: 1,
  };
}

export const useLiveSummary = (league: string, eventId: string, enabled = true) => useQuery(summaryQuery(league, eventId, enabled));

/** Live scores for one ET date ("YYYY-MM-DD"); off for future dates. Empty map while unavailable. */
export function useLiveScoreboard(league: string, date: string | undefined) {
  const query = useQuery({
    queryKey: ["espn-scoreboard", league, date],
    queryFn: async ({ signal }) => parseScoreboard(league, await espn(league, "scoreboard", { dates: date!.replaceAll("-", "") }, signal)),
    enabled: !!date && /^\d{4}-\d{2}-\d{2}$/.test(date) && date <= easternToday(),
    refetchInterval: ({ state }) => state.error ? BACKOFF : scoreboardPoll(Object.values(state.data ?? {})),
    staleTime: 10_000,
    retry: 1,
  });
  return query.data ?? ({} as Record<string, LiveGame>);
}
