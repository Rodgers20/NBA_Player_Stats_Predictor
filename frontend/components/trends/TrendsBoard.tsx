"use client";

import { useState } from "react";
import Image from "next/image";
import { PlayerName } from "@/components/analysis/PlayerName";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { fmtOdds } from "@/lib/utils";
import { formatOdds } from "@/lib/prop-research.mjs";
import { TrendSlipButton } from "@/components/trends/TrendSlipButton";
import type { TrendCard } from "@/lib/types";

const PAGE = 24;   // 330 streak cards rendered at once made a 26,000px page
const SIDES = [["all", "All"], ["over", "Overs"], ["under", "Unders"]] as const;
const STATS = [
  ["all", "All"], ["PTS", "PTS"], ["REB", "REB"], ["AST", "AST"], ["FG3M", "3PM"], ["STL", "STL"], ["BLK", "BLK"],
  ["PTS+REB", "P+R"], ["PTS+AST", "P+A"], ["REB+AST", "R+A"], ["PTS+REB+AST", "PRA"],
] as const;

const STAT_LABEL: Record<string, string> = { FG3M: "3PM", "PTS+REB": "P+R", "PTS+AST": "P+A", "REB+AST": "R+A", "PTS+REB+AST": "PRA" };

/** Each of the last games as a bar of its real value against a dashed line at the betting line: the cushion is visible, not just a tick. */
function GameBars({ card }: { card: TrendCard }) {
  const values = card.values ?? [];
  if (!values.length) return null;
  const top = Math.max(...values, card.line) * 1.12 || 1;
  const linePct = (card.line / top) * 100;
  return <figure className="m-0 pr-9" aria-label={`Last ${values.length} games, oldest to newest: ${values.join(", ")} against a ${card.line} line`}>
    <div className="relative h-16">
      <div className="absolute inset-x-0 border-t border-dashed border-white/45" style={{ bottom: `${linePct}%` }} aria-hidden="true">
        {/* The line's value sits in a right gutter so it never covers the latest game's bar. */}
        <span className="absolute -right-9 -top-2 w-8 text-center font-mono text-[10px] font-semibold text-white/75">{card.line}</span>
      </div>
      <div className="flex h-full items-end gap-1" aria-hidden="true">
        {values.map((value, index) => <span key={index} title={String(value)}
          className={`flex-1 rounded-t-[3px] ${card.results[index] ? "bg-teal-400/85" : "bg-rose-400/75"}`}
          style={{ height: `${Math.max(4, (value / top) * 100)}%` }} />)}
      </div>
    </div>
    <div className="mt-1 flex gap-1 font-mono text-[10px] tabular-nums text-text-sec" aria-hidden="true">
      {values.map((value, index) => <span key={index} className="flex-1 text-center">{value}</span>)}
    </div>
    {/* Hit or miss against the prop line, game by game: the strip people scan first. */}
    <div className="mt-2 flex gap-1" role="img"
      aria-label={`Against ${card.side} ${card.line}, oldest to newest: ${card.results.map(hit => hit ? "hit" : "miss").join(", ")}`}>
      {card.results.map((hit, index) => <span key={index} className={`h-2 flex-1 rounded-full ${hit ? "bg-teal-400" : "bg-rose-400"}`} />)}
    </div>
  </figure>;
}

function Headshot({ card }: { card: TrendCard }) {
  const [failed, setFailed] = useState(false);
  const initial = card.player.trim().split(/\s+/).at(-1)?.[0]?.toUpperCase() || "?";
  return <div className="relative shrink-0">
    <div className="flex h-14 w-14 items-center justify-center overflow-hidden rounded-full border border-white/10 bg-[#59e0c8]/10 text-lg font-bold text-[#59e0c8]">
      {card.headshot_url && !failed
        ? <Image src={card.headshot_url} alt="" width={56} height={56} unoptimized onError={() => setFailed(true)} className="h-14 w-14 object-cover object-top" />
        : initial}
    </div>
    <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 rounded bg-[#0f171c] px-1.5 font-mono text-[9px] font-bold text-teal-300 ring-1 ring-white/10">{card.team}</span>
  </div>;
}

function TrendItem({ card, league }: { card: TrendCard; league: "nba" | "wnba" }) {
  const stat = STAT_LABEL[card.stat] ?? card.stat;
  const side = card.side === "over" ? "Over" : "Under";
  const margin = Math.abs(card.avg - card.line).toFixed(1);
  const perfect = card.hits === card.games;
  return <li className="ql-panel flex flex-col gap-4 p-4">
    <div className="flex items-center gap-3">
      <Headshot card={card} />
      <div className="min-w-0 flex-1">
        <PlayerName name={card.player} league={league} className="block max-w-full truncate font-bold text-text-pri" />
        <p className="text-xs text-text-sec">{card.opponent ? `vs ${card.opponent}` : card.team}{card.playing_today ? " · tonight" : ""}</p>
        <p className="mt-1 text-lg font-extrabold leading-none text-text-pri">{side} {card.line} <span className="text-[#59e0c8]">{stat}</span></p>
      </div>
      <div className={`shrink-0 rounded-lg px-2.5 py-1.5 text-center ${perfect ? "bg-teal-400 text-[#0d151b]" : "border border-teal-400/40 text-teal-300"}`}>
        <p className="font-mono text-lg font-bold leading-none tabular-nums">{card.hits}/{card.games}</p>
        <p className="mt-0.5 text-[9px] font-bold uppercase tracking-wider">{perfect ? "perfect" : "hit"}</p>
      </div>
    </div>
    <GameBars card={card} />
    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.07] pt-3 text-xs text-text-sec">
      <span className="tabular-nums">avg <strong className="text-text-pri">{card.avg}</strong> · <strong className="text-teal-300">{margin}</strong> {card.side === "over" ? "over" : "under"} the line</span>
      {card.live
        ? <span className="flex items-center gap-2">
            <span className="font-mono tabular-nums text-text-pri">{formatOdds(card.live.price) ?? fmtOdds(card.live.price)}</span>
            {card.live.book && <span className="uppercase tracking-wide">{card.live.book}</span>}
            {/* Only a sportsbook line can go on a slip; a research line has no price to bet. */}
            <TrendSlipButton card={card} league={league} />
          </span>
        : <span className="ql-data-label">RESEARCH LINE · NO PRICE</span>}
    </div>
  </li>;
}

/** Players clearing the same line in at least eight of their last ten games: the Props page's "Streaks" view. */
export function TrendsBoard({ league }: { league: "nba" | "wnba" }) {
  const [side, setSide] = useState<(typeof SIDES)[number][0]>("all");
  const [stat, setStat] = useState<(typeof STATS)[number][0]>("all");
  const [scope, setScope] = useState<"tonight" | "all">("tonight");
  const [team, setTeam] = useState("");
  const [shown, setShown] = useState(PAGE);
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["trends", league], queryFn: () => api.trends(league, { limit: 400 }), staleTime: 300_000 });
  const slateTeams = data?.slate?.teams ?? [];
  // Without a known slate there is nothing to narrow to, so show everyone rather than an empty board.
  const tonight = scope === "tonight" && slateTeams.length > 0;
  const cards = (data?.trends ?? []).filter(card => (side === "all" || card.side === side) && (stat === "all" || card.stat === stat)
    && (!tonight || (card.playing_today === true && (!team || card.team === team))));
  const chip = (active: boolean) => `ql-chip shrink-0 ${active ? "ql-chip-active" : ""}`;

  return <>
    <p className="mb-3 text-xs text-text-sec">Players clearing the same line in at least eight of their last ten games, biggest cushion first. History is not a forecast.</p>
    {slateTeams.length > 0 && <div className="mb-3 flex flex-wrap items-center gap-3">
      <div className="flex gap-1.5" role="group" aria-label="Players shown">
        <button type="button" aria-pressed={tonight} onClick={() => setScope("tonight")} className={chip(tonight)}>Tonight&apos;s games</button>
        <button type="button" aria-pressed={!tonight} onClick={() => setScope("all")} className={chip(!tonight)}>All players</button>
      </div>
      {tonight && <label className="ql-data-label flex items-center gap-2">TEAM
        <select value={team} onChange={event => setTeam(event.target.value)} className="ql-control min-h-8 px-2.5 py-1.5 text-xs font-medium normal-case tracking-normal text-text-pri">
          <option value="">All {slateTeams.length} teams playing</option>
          {slateTeams.map(abbrev => <option key={abbrev} value={abbrev}>{abbrev}</option>)}
        </select>
      </label>}
    </div>}

    <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-y border-white/10 py-3">
      <div className="flex max-w-full gap-1.5 overflow-x-auto pb-1" role="group" aria-label="Side">{SIDES.map(([value, label]) => <button key={value} type="button" aria-pressed={side === value} onClick={() => setSide(value)} className={chip(side === value)}>{label}</button>)}</div>
      <div className="flex max-w-full gap-1.5 overflow-x-auto pb-1" role="group" aria-label="Statistic">{STATS.map(([value, label]) => <button key={value} type="button" aria-pressed={stat === value} onClick={() => setStat(value)} className={chip(stat === value)}>{label}</button>)}</div>
    </div>

    {isPending && <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3" aria-label="Loading streaks">{[0, 1, 2].map(index => <div key={index} className="ql-panel h-40 animate-pulse" />)}</div>}
    {isError && <div role="alert" className="ql-panel p-5 text-sm text-red-300">Streaks are unavailable. <button type="button" onClick={() => void refetch()} className="underline">Try again</button></div>}
    {!isPending && !isError && cards.length === 0 && <div role="status" className="ql-panel p-6">
      <h2 className="ql-section-title">{data?.trends.length ? "No streaks match these filters" : "No qualifying streaks right now"}</h2>
      <p className="mt-2 max-w-2xl text-sm text-text-sec">{data?.trends.length ? (tonight ? "No streaks among tonight's players match these filters. Try another side, stat or team, or switch to All players." : "Try another side or stat.") : "A streak needs a player with ten games inside the last 14 days, so this fills in once the season is underway."}</p>
    </div>}
    <ul className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{cards.slice(0, shown).map(card => <TrendItem key={`${card.player}-${card.stat}-${card.side}`} card={card} league={league} />)}</ul>
    {cards.length > shown && <button type="button" onClick={() => setShown(count => count + PAGE)}
      className="ql-control mt-4 w-full py-3 text-sm font-semibold text-[#59e0c8] hover:bg-[#59e0c8]/10">Show more ({cards.length - shown} remaining)</button>}
  </>;
}
