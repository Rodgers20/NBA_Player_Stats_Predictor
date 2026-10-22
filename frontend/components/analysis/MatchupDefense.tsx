"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { DEF_STATS, basePosition, positionMatchup } from "@/lib/defense-view.mjs";
import type { PositionCell } from "@/lib/defense-view.mjs";
import type { Game } from "@/lib/types";

const TONE: Record<string, string> = {
  A: "border-teal-400/30 bg-teal-400/10 text-teal-400",
  B: "border-teal-400/30 bg-teal-400/10 text-teal-400",
  C: "border-white/15 bg-white/5 text-text-sec",
  D: "border-rose-400/25 bg-rose-400/10 text-rose-400",
  F: "border-rose-400/25 bg-rose-400/10 text-rose-400",
};
const ordinal = (n: number) => { const s = ["th", "st", "nd", "rd"], v = n % 100; return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`; };

function Grade({ grade }: { grade: string }) {
  return <span className={`rounded border px-1.5 py-0.5 text-[10px] font-bold ${TONE[grade]}`}>{grade}</span>;
}

/** Allowed average against the league mean for the same position, drawn like the Season Trends bars. */
function AllowedBar({ cell, color }: { cell: PositionCell; color: string }) {
  const max = Math.max(cell.allowed, cell.leagueAvg, 1);
  return <div className="space-y-1.5" role="img" aria-label={`Allowed ${cell.allowed.toFixed(1)}, league average ${cell.leagueAvg.toFixed(1)}`}>
    <div className="h-2 bg-white/5"><div className="h-2" style={{ width: `${cell.allowed / max * 100}%`, background: color }} /></div>
    <div className="h-2 bg-white/5"><div className="h-2 bg-slate-500" style={{ width: `${cell.leagueAvg / max * 100}%` }} /></div>
  </div>;
}

export function MatchupDefense({ league, opponent, game, position, stat, accent }: {
  league: "nba" | "wnba"; opponent: string; game: Game; position: string | undefined; stat: string; accent: string;
}) {
  const defense = useQuery({ queryKey: ["defense", league], queryFn: () => api.defense(league), staleTime: 300_000 });
  const own = basePosition(position);
  const [chosen, setChosen] = useState<"G" | "F" | "C" | null>(null);
  const slot = chosen ?? own;   // the data has three position groups, so the tabs are G, F and C
  const matchup = slot && defense.data ? positionMatchup(defense.data.rankings, opponent, slot) : {};
  const headline = DEF_STATS.includes(stat as never) ? matchup[stat as keyof typeof matchup] : undefined;
  const involved = stat.split("+");
  const homeSide = game.home_team === opponent;
  const injuries = (homeSide ? game.home_injuries : game.away_injuries) ?? [];
  const wins = homeSide ? game.home_wins : game.away_wins;
  const losses = homeSide ? game.home_losses : game.away_losses;
  const flagged = injuries.filter(item => /out|doubt|quest/i.test(item.status)).slice(0, 4);

  return <div className="mt-4 border-t border-white/10 pt-4">
    <div className="flex items-baseline justify-between gap-2">
      <span className="ql-data-label">{opponent} DEFENSE{slot ? ` VS ${slot}` : ""}</span>
      {wins != null && losses != null && <span className="ql-data-label">{wins}–{losses}</span>}
    </div>
    {defense.data && <div className="mt-2 flex gap-1" role="group" aria-label="Defense against position">
      {(["G", "F", "C"] as const).map(option => <button key={option} type="button" aria-pressed={slot === option} onClick={() => setChosen(option)}
        className={`ql-chip min-h-7 px-2.5 py-0.5 text-[11px] font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8] ${slot === option ? "ql-chip-active" : "text-[#a9bec0]"}`}>
        vs {option}{own === option ? " · own" : ""}</button>)}
    </div>}
    {defense.isPending && <div className="mt-3 h-16 animate-pulse bg-white/5" />}
    {defense.isError && <p className="mt-3 text-xs text-text-sec">Defense rankings are unavailable.</p>}
    {defense.data && !slot && <p className="mt-3 text-xs text-text-sec">Player position unknown, so a position matchup can&apos;t be shown.</p>}
    {defense.data && slot && !Object.keys(matchup).length && <p className="mt-3 text-xs text-text-sec">No defense-vs-position data for {opponent} yet.</p>}
    {headline && <div className="mt-3">
      <div className="mb-2 flex items-center justify-between gap-2 text-xs">
        <span className="text-text-sec"><strong className="text-text-pri">{headline.allowed.toFixed(1)} {stat}</strong> allowed · league {headline.leagueAvg.toFixed(1)}</span>
        <Grade grade={headline.grade} />
      </div>
      <AllowedBar cell={headline} color={accent} />
      <p className="mt-2 text-[11px] text-text-sec">{ordinal(headline.rank)} softest of {headline.of} · <span style={{ color: accent }}>●</span> {opponent} <span className="text-slate-400">●</span> league average</p>
    </div>}
    {Object.keys(matchup).length > 0 && <div className="mt-3 grid grid-cols-3 gap-px overflow-hidden border border-white/10 bg-white/10">
      {DEF_STATS.map(key => {
        const cell = matchup[key as keyof typeof matchup];
        return <div key={key} className={`bg-[#152128] p-2 ${involved.includes(key) ? "" : "opacity-60"}`}>
          <p className="ql-data-label">{key}</p>
          {cell ? <p className="mt-1 flex items-center gap-1.5 text-sm font-bold tabular-nums text-text-pri">{cell.rank}<span className="text-[10px] font-normal text-text-sec">/{cell.of}</span><Grade grade={cell.grade} /></p>
            : <p className="mt-1 text-sm text-text-sec">—</p>}
        </div>;
      })}
    </div>}
    {Object.keys(matchup).length > 0 && <p className="mt-2 text-[11px] text-text-sec">Rank 1 allows the most to {slot}s; grade A is the softest matchup.</p>}
    <p className="ql-data-label mt-4">{opponent} AVAILABILITY</p>
    {flagged.length
      ? <ul className="mt-2 space-y-1 text-xs text-text-sec">{flagged.map((item, index) => <li key={`${item.name}-${index}`} className="flex justify-between gap-2"><span className="truncate text-text-pri">{item.name}</span><span className="shrink-0 font-mono text-amber-300">{item.status}</span></li>)}</ul>
      : <p className="mt-2 text-xs text-text-sec">No injuries reported for {opponent}.</p>}
  </div>;
}
