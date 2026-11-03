"use client";
import Image from "next/image";
import { useState } from "react";
import type { PlayerDetails, PlayerProfile } from "@/lib/types";

export function PlayerCareer({ profile, details, loading, error }: {
  profile: PlayerProfile; details?: PlayerDetails; loading: boolean; error: boolean;
}) {
  const [tab, setTab] = useState<"teams" | "competitions">("teams");
  return <section className="overflow-hidden rounded-2xl border border-white/[.07] bg-[#1b1e23]">
    <h2 className="border-b border-white/[.07] px-6 py-5 text-base font-semibold text-ink">Career</h2>
    <div role="tablist" aria-label="Career view" className="flex gap-7 border-b border-white/[.07] px-6">
      {([['teams','Teams'],['competitions','Competition records']] as const).map(([key,label]) =>
        <button key={key} type="button" role="tab" aria-selected={tab===key} onClick={()=>setTab(key)}
          className={`min-h-12 border-b-2 text-xs transition-colors ${tab===key?'border-pitch font-semibold text-ink':'border-transparent text-ink-muted hover:text-ink'}`}>{label}</button>)}
    </div>
    {tab==='teams' ? <div className="px-6 pb-5">
      <p className="pb-1 pt-5 text-xs text-ink-muted">Recorded teams & seasons</p>
      {loading && <p role="status" className="py-5 text-sm text-ink-muted">Loading career records…</p>}
      {details?.career.map(team => <div key={team.team_id} className="flex gap-3 border-b border-white/[.06] py-4 last:border-0">
        {team.crest ? <Image unoptimized src={team.crest} alt="" width={26} height={26} className="mt-1 h-7 w-7 object-contain" /> : <span className="h-7 w-7 rounded-full bg-elevated" />}
        <div className="min-w-0"><p className="text-sm text-ink">{team.team}</p>
          <p className="mt-1 text-xs leading-relaxed text-ink-muted">{team.seasons.length ? team.seasons.join(' · ') : 'Season dates unavailable'}</p>
        </div>
      </div>)}
      {!loading && !details?.career.length && <p className="py-5 text-sm leading-relaxed text-ink-muted">{error ? 'Career records could not be loaded. Try again later.' : 'Career records are not available for this player.'}</p>}
      {!!details?.career.length && <p className="mt-4 text-[10px] leading-relaxed text-ink-muted">{details.source} · Recorded seasons, not contract dates. Career appearance and goal totals are not supplied.</p>}
    </div> : <div className="px-6 py-4">
      <div className="grid grid-cols-[1fr_42px_42px] gap-2 text-[10px] uppercase tracking-wider text-ink-muted"><span>Competition</span><span className="text-right">Apps</span><span className="text-right">Goals</span></div>
      {profile.spells.map((spell,i) => <div key={i} className="grid grid-cols-[1fr_42px_42px] items-center gap-2 border-b border-white/[.06] py-4 last:border-0">
        <div className="min-w-0"><p className="text-sm text-ink">{spell.represented ?? spell.competition}</p><p className="mt-1 text-xs text-ink-muted">{spell.competition}{spell.season ? ` · ${spell.season}` : ""}</p></div>
        <span className="text-right text-sm tabular-nums text-ink-secondary">{spell.stats.matches ?? spell.stats.played ?? '—'}</span>
        <span className="text-right text-sm tabular-nums text-ink-secondary">{spell.stats.goals ?? '—'}</span>
      </div>)}
      <p className="mt-4 text-[10px] text-ink-muted">Available competition records; not career totals.</p>
    </div>}
  </section>;
}
