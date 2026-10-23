"use client";

import { useState } from "react";
import Image from "next/image";
import { PlayerName, focusFromProp } from "@/components/analysis/PlayerName";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatOdds } from "@/lib/prop-research.mjs";
import { gameInsights } from "@/lib/insights.mjs";
import { rowDashes } from "@/lib/prop-rows.mjs";
import { HitDashes } from "@/components/ui/HitDashes";
import type { Game, Prop } from "@/lib/types";

function InsightCard({ prop, text, league }: { prop: Prop; text: string; league: "nba" | "wnba" }) {
  const [photoFailed, setPhotoFailed] = useState(false);
  const side = /^under$/i.test(prop.direction) ? "Under" : "Over";
  const initial = prop.player.trim().split(/\s+/).at(-1)?.[0]?.toUpperCase() || "?";
  return (
    <li className="flex flex-wrap items-center gap-x-4 gap-y-3 border-b border-white/[0.07] px-4 py-4 last:border-b-0 sm:px-5">
      <div className="flex h-11 w-11 shrink-0 items-center justify-center overflow-hidden rounded-md border border-white/10 bg-[#59e0c8]/10 text-sm font-bold text-[#59e0c8]" aria-hidden="true">
        {prop.headshot_url && !photoFailed
          ? <Image src={prop.headshot_url} alt="" width={44} height={44} unoptimized onError={() => setPhotoFailed(true)} className="h-11 w-11 object-cover object-top" />
          : initial}
      </div>
      <div className="min-w-0 flex-1 basis-64">
        <p className="text-sm text-text-pri">{text}</p>
        <HitDashes dashes={rowDashes(prop)} wide />
      </div>
      <div className="flex items-center gap-3 text-right">
        <div>
          <p className="inline-block rounded-md border border-white/10 bg-white/[0.06] px-2 py-1 font-mono text-sm font-semibold tabular-nums text-text-pri">{side} {prop.line} {formatOdds(prop.price)}</p>
          <p className="mt-1 text-[10px] uppercase tracking-wide text-text-sec">{prop.book ?? "Sportsbook"}</p>
        </div>
        <PlayerName name={prop.player} league={league} focus={focusFromProp(prop)} className="text-xs font-bold text-[#59e0c8] hover:underline">Analyze →</PlayerName>
      </div>
    </li>
  );
}

/** Streaks worth knowing about for this game's priced props: written from the last ten games, ranked by how long or how often. */
export function GameInsights({ game, league }: { game: Game; league: "nba" | "wnba" }) {
  const props = useQuery({
    queryKey: ["game-props", league, game.matchup],
    queryFn: () => api.props({ league, game: game.matchup, include_research: true, sort: "ev", limit: 100 }),
    staleTime: 120_000,
  });
  if (props.isPending) return <div className="ql-panel h-48 animate-pulse" aria-busy="true" aria-label="Loading insights" />;
  if (props.isError) return <div role="alert" className="ql-panel p-5 text-sm text-text-sec">This game&apos;s insights could not be loaded. <button type="button" className="underline" onClick={() => void props.refetch()}>Try again</button></div>;

  const insights = gameInsights(props.data.props);
  if (!insights.length) return (
    <div role="status" className="ql-panel p-5 text-sm text-text-sec">
      <p className="font-bold text-text-pri">No streaks to report for {game.matchup} yet.</p>
      <p className="mt-1">{props.data.message || "Insights come from priced props with at least five recent games over or under the line. They appear once sportsbooks post lines."}</p>
    </div>
  );
  return (
    <section aria-label={`${game.matchup} insights`}>
      <ul className="ql-panel overflow-hidden">{insights.map(({ prop, text }) => <InsightCard key={`${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`} prop={prop} text={text} league={league} />)}</ul>
      <p className="mt-2 text-[11px] text-text-sec">Past results against today&apos;s lines: descriptive, not a forecast. Prices worse than -500 are left out.</p>
    </section>
  );
}
