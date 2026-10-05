"use client";

import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import { usePrefs } from "@/store/prefs";
import type { Prop } from "@/lib/types";

interface Props {
  prop: Prop;
  targetDate?: string | null;
  onAnalyze: (prop: Prop) => void;
  selected?: boolean;
}

function odds(value: number | null | undefined) {
  if (value == null) return null;
  return value > 0 ? `+${Math.round(value)}` : `${Math.round(value)}`;
}

export function PropCard({ prop, targetDate, onAnalyze, selected = false }: Props) {
  const [photoFailed, setPhotoFailed] = useState(false);
  const league = usePrefs((state) => state.league);
  const initial = prop.player.trim().split(/\s+/).at(-1)?.[0]?.toUpperCase() || "?";
  const hitRate = prop.hit_rate != null && Number.isFinite(prop.hit_rate) ? Math.round(prop.hit_rate) : null;
  const projectionOnly = prop.line == null && prop.model_projection != null;
  const quotedSide = prop.direction.toLowerCase() === "under" ? "Under" : "Over";
  const verified = prop.recommendation_eligible ?? prop.has_live_odds;
  const ev = !verified || prop.ev == null ? "—" : `${prop.ev >= 0 ? "+" : ""}${(prop.ev * 100).toFixed(1)}%`;
  const trackParams = new URLSearchParams({ league, player: prop.player, stat: prop.stat, side: quotedSide });
  if (targetDate) trackParams.set("game_date", targetDate);
  if (prop.line != null) trackParams.set("line", String(prop.line));
  if (verified && prop.price != null) trackParams.set("price", String(prop.price));

  return (
    <article className={`border-b border-white/[0.07] px-4 py-3 transition-colors last:border-b-0 sm:px-5 ${selected ? "bg-[#59e0c8]/[0.07] shadow-[inset_3px_0_0_#59e0c8]" : "hover:bg-white/[0.025]"}`}>
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-3 md:grid-cols-[minmax(0,1.5fr)_minmax(8rem,1fr)_5rem_4.5rem_auto]">
        <div className="flex min-w-0 items-center gap-3">
          <button type="button" onClick={() => onAnalyze(prop)} aria-label={`Preview ${prop.player} history`} className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-md border border-white/10 bg-[#59e0c8]/10 text-sm font-bold text-[#59e0c8] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">
            {prop.headshot_url && !photoFailed
              ? <Image src={prop.headshot_url} alt="" width={40} height={40} unoptimized onError={() => setPhotoFailed(true)} className="h-10 w-10 object-cover object-top" />
              : initial}
          </button>
          <div className="min-w-0">
            <button type="button" onClick={() => onAnalyze(prop)} className="block max-w-full truncate text-left text-sm font-bold text-[#eaf8f6] hover:text-[#59e0c8] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">{prop.player}</button>
            <p className="mt-0.5 truncate text-[11px] text-[#a9bec0]">{prop.team} · {prop.stat_label || prop.stat}</p>
          </div>
        </div>
        <div className="text-right md:text-left">
          <p className="whitespace-nowrap text-sm font-bold text-[#59e0c8]">{projectionOnly ? `${prop.model_projection?.toFixed(1)} ${prop.stat}` : `${quotedSide} ${prop.line ?? "—"}`} {verified && prop.has_live_odds && prop.price != null && <span className="ml-1 text-[#eaf8f6]">{odds(prop.price)}</span>}</p>
          <p className={`ql-data-label mt-0.5 text-[10px] ${verified && prop.has_live_odds ? "text-[#f5ba64]" : "text-[#a9bec0]"}`}>{projectionOnly ? "MODEL PROJECTION · NO LINE" : verified && prop.has_live_odds ? "SPORTSBOOK QUOTE" : "HISTORICAL LINE"}</p>
        </div>
        <div className="hidden text-right md:block" title="Historical hit rate">
          <p className="text-sm font-bold tabular-nums text-[#eaf8f6]">{hitRate == null ? "—" : `${hitRate}%`}</p>
          <p className="ql-data-label text-[10px] text-[#a9bec0]">HIT RATE</p>
        </div>
        <div className="hidden text-right md:block" title="Expected value for verified markets only">
          <p className="text-sm font-bold tabular-nums text-[#eaf8f6]">{ev}</p>
          <p className="ql-data-label text-[10px] text-[#a9bec0]">EV</p>
        </div>
        <div className="hidden items-center gap-2 md:flex">
          <button type="button" onClick={() => onAnalyze(prop)} className="ql-control whitespace-nowrap px-2.5 py-1.5 text-xs font-semibold text-[#d0e0df] hover:border-[#59e0c8]/50 hover:text-[#59e0c8] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">Inspect</button>
          {!projectionOnly && <Link href={`/bets?${trackParams}`} className="ql-control whitespace-nowrap px-2.5 py-1.5 text-xs font-semibold text-[#59e0c8] hover:border-[#59e0c8]/50 hover:bg-[#59e0c8]/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]">Track</Link>}
        </div>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 pl-[52px] text-[11px] text-[#a9bec0]">
        <span className="truncate">{prop.game_matchup || `${prop.team} vs ${prop.opponent}`}</span>
        {prop.avg != null && <span>AVG <strong className="tabular-nums text-[#eaf8f6]">{prop.avg}</strong></span>}
        {prop.is_combo && <span className="ql-chip px-1.5 py-0.5 text-[10px]">COMBO</span>}
        {prop.blowout_risk && <span className="rounded border border-orange-400/40 px-1.5 py-0.5 text-[10px] font-bold text-orange-300">BLOWOUT RISK</span>}
        {!verified && <span className="rounded border border-amber-400/40 bg-amber-400/[0.07] px-1.5 py-0.5 text-[10px] font-bold text-amber-300">RESEARCH ONLY</span>}
      </div>
      {prop.insight && <p className="mt-2 pl-[52px] text-xs leading-relaxed text-[#a9bec0]">{prop.insight}</p>}
      {prop.quality_reason && <p className="mt-1 pl-[52px] text-xs text-amber-300">{prop.quality_reason}</p>}
      <div className="mt-2 flex gap-4 pl-[52px] text-xs md:hidden">
        <span className="text-[#a9bec0]">Hit <strong className="tabular-nums text-[#eaf8f6]">{hitRate == null ? "—" : `${hitRate}%`}</strong></span>
        <span className="text-[#a9bec0]">EV <strong className="tabular-nums text-[#eaf8f6]">{ev}</strong></span>
        {!projectionOnly && <Link href={`/bets?${trackParams}`} className="ml-auto font-semibold text-[#59e0c8]">Track pick →</Link>}
      </div>
    </article>
  );
}
