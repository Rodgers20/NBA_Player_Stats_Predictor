"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { SeasonXi } from "./SeasonXi";

export function LatestLineup({ team, code }: { team: string; code: string }) {
  const query = useQuery({ queryKey: ["latest-lineup-season-ratings", team, code], queryFn: () => api.latestLineup(team, code), staleTime: 15 * 60 * 1000 });
  if (query.isPending) return <p role="status" className="p-5 text-sm text-ink-muted">Loading confirmed starters…</p>;
  if (query.isError) return <div className="p-5 text-sm text-ink-muted">Could not load the last lineup. <button className="text-pitch underline" onClick={() => query.refetch()}>Retry</button></div>;
  const data = query.data;
  return <div>
    <p className="mb-3 text-sm font-semibold text-ink">{data.stats_competition} {data.stats_season} · Season stats</p>
    {data.match_label && <div className="mb-4 px-1">
      <p className="text-sm font-semibold text-ink">{data.match_label}</p>
      <p className="mt-1 text-xs text-ink-muted">{data.match_date?.slice(0, 10)} · {data.competition}{data.formation ? ` · ${data.formation}` : ""}</p>
    </div>}
    <SeasonXi data={data} />
    {data.ratings_note && <a href={data.ratings_source_url} target="_blank" rel="noreferrer" className="mt-3 block text-xs text-pitch underline">{data.ratings_note}</a>}
    {data.stats_note && <p className="mt-2 text-xs text-ink-muted">{data.stats_note}</p>}
    {data.source_url && <a href={data.source_url} target="_blank" rel="noreferrer" className="mt-3 inline-block text-xs text-pitch underline">View match source ↗</a>}
  </div>;
}
