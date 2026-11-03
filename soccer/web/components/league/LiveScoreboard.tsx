"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { API_BASE } from "@/lib/api";
import { TeamLabel } from "@/components/TeamCrest";
import type { LiveMatch } from "@/lib/types";

type Score = LiveMatch & { is_live: boolean; completed: boolean; status_label: string; source_url: string; stale: boolean; updated_at: string | null };
type Board = { matches: Score[]; stale: boolean; competitions: { updated_at: string | null; stale: boolean }[] };

export function LiveScoreboard({ code }: { code: string }) {
  const router = useRouter();
  const query = useQuery({queryKey:["espn-live",code], queryFn:async()=>{
    const response=await fetch(`${API_BASE}/live/scores?competition_code=${encodeURIComponent(code)}`);
    if (!response.ok) throw new Error("Score refresh failed");
    const body=await response.json(); return body.data as Board;
  },refetchInterval:45000,refetchIntervalInBackground:false,staleTime:30000,retry:1});
  useEffect(()=>{const timer=setInterval(()=>{if(document.visibilityState==="visible") router.refresh();},45000);return()=>clearInterval(timer);},[router]);
  if(query.isPending) return <p role="status" className="p-4 text-sm text-ink-muted">Loading ESPN scores…</p>;
  const data=query.data;
  const stale=query.isError || data?.stale;
  return <div className="space-y-3">
    <p role="status" className={`text-xs ${stale?"text-amber-400":"text-ink-muted"}`}>
      {stale?"Scores may be out of date. Refresh unavailable.":"ESPN · Refreshes every 45 seconds during play"}
      {data?.competitions[0]?.updated_at && <span className="block mt-1">Last updated {new Date(data.competitions[0].updated_at).toLocaleTimeString()}</span>}
    </p>
    {query.isError && <button onClick={()=>query.refetch()} className="text-xs text-pitch underline">Retry</button>}
    {!data?.matches.length && <p className="py-4 text-sm text-ink-muted">{stale?"Match availability could not be confirmed.":"No matches scheduled today (UTC)."}</p>}
    <ul className="space-y-2">{data?.matches.map(m=><li key={m.fixture_key}>
      <a href={m.source_url} target="_blank" rel="noreferrer" className="block rounded-lg bg-elevated px-3 py-3 hover:bg-white/5">
        <div className="mb-2 flex justify-between text-[11px] text-ink-muted"><span className={m.is_live&&!stale?"text-live font-semibold":""}>{stale?"Last known: ":""}{m.status_label}</span><span>{m.is_live&&!stale?"LIVE":m.completed?"Full time":m.time} · ESPN ↗</span></div>
        <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2">
          <TeamLabel name={m.home_team} size={20} className="min-w-0 text-xs" />
          <strong className="whitespace-nowrap font-mono text-lg">{m.home_goals ?? "—"} – {m.away_goals ?? "—"}</strong>
          <TeamLabel name={m.away_team} size={20} className="min-w-0 text-xs" />
        </div>
      </a>
    </li>)}</ul>
  </div>;
}
