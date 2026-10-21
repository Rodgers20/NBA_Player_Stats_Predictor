"use client";

import { Check, Plus } from "lucide-react";
import { useSlip } from "@/store/slip";
import { legKey } from "@/lib/bet-slip.mjs";
import { EDGE_MIN, bettableEdges, type GameEdge } from "@/lib/game-edges.mjs";
import { fmtPrice } from "@/lib/game-markets.mjs";
import { mlLeg } from "@/lib/team-legs.mjs";

const pct = (value: number) => `${Math.round(value * 100)}%`;
const evText = (ev: number) => `${ev >= 0 ? "+" : ""}${(ev * 100).toFixed(1)}%`;

function EdgeRow({ edge, league, date }: { edge: GameEdge; league: "nba" | "wnba"; date: string }) {
  const legs = useSlip(state => state.legs);
  const toggle = useSlip(state => state.toggle);
  const leg = date ? mlLeg({ game: edge.game, side: edge.side, league, date, probability: edge.modelProb }) : null;
  const inSlip = leg ? legs.some(item => legKey(item) === legKey(leg)) : false;
  return (
    <li className="flex items-center gap-3 border-b border-white/[0.07] px-4 py-3 last:border-b-0">
      <button type="button" disabled={!leg} aria-pressed={inSlip} onClick={() => leg && toggle({ ...leg, addedAt: new Date().toISOString() })}
        aria-label={`${inSlip ? "Remove" : "Add"} ${edge.team} moneyline ${fmtPrice(edge.price)} ${inSlip ? "from" : "to"} slip`}
        className={`grid h-9 w-9 shrink-0 place-items-center rounded-full border transition-colors disabled:opacity-30 ${inSlip ? "border-[#59e0c8] bg-[#59e0c8] text-[#0d151b]" : "border-[#59e0c8]/45 text-[#59e0c8] hover:bg-[#59e0c8]/10"}`}>
        {inSlip ? <Check size={16} aria-hidden="true" /> : <Plus size={16} aria-hidden="true" />}
      </button>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-bold text-[#eaf8f6]">{edge.team} moneyline <span className="ml-1 font-mono text-[#f5ba64]">{fmtPrice(edge.price)}</span></p>
        <p className="mt-0.5 text-[11px] text-[#a9bec0]">{edge.game.matchup} · model {pct(edge.modelProb)}{edge.marketProb != null && <> vs market {pct(edge.marketProb)}</>}</p>
        {edge.ev > 0.1 && <p className="mt-0.5 text-[11px] leading-snug text-[#f5ba64]">Large gap: check injuries and lineups first. A gap this big is more often a missing input than a mispriced line.</p>}
      </div>
      <p className={`shrink-0 text-right font-display text-2xl font-bold tabular-nums ${edge.ev >= EDGE_MIN ? "text-[#59e0c8]" : "text-[#a9bec0]"}`}>{evText(edge.ev)}<span className="block font-mono text-[10px] font-medium text-[#a9bec0]">EV / $1</span></p>
    </li>
  );
}

/** Moneyline value from the game model versus the sportsbook price, for tonight's games that have both. */
export function GameEdges({ edges, league, date }: { edges: GameEdge[]; league: "nba" | "wnba"; date: string }) {
  if (!edges.length) return null;
  const good = bettableEdges(edges);
  const rest = edges.filter(edge => !good.includes(edge));
  return (
    <section aria-label="Game line edges" className="mb-6">
      <div className="mb-2 flex items-baseline justify-between border-b border-white/10 pb-2">
        <h2 className="ql-section-title">Game line edges</h2>
        <span className="ql-data-label">{good.length} {good.length === 1 ? "EDGE" : "EDGES"} · MONEYLINE</span>
      </div>
      {good.length > 0
        ? <ul className="ql-panel overflow-hidden">{good.map(edge => <EdgeRow key={`${edge.game.matchup}-${edge.side}`} edge={edge} league={league} date={date} />)}</ul>
        : <p className="ql-panel px-4 py-4 text-sm text-text-sec" role="status">No moneyline clears the {Math.round(EDGE_MIN * 100)}% edge bar: the model&apos;s win chances sit close to the sportsbook&apos;s.</p>}
      {rest.length > 0 && (
        <details className="mt-3">
          <summary className="ql-data-label cursor-pointer py-1">PRICED, NO EDGE · {rest.length}</summary>
          <ul className="ql-panel mt-2 overflow-hidden">{rest.map(edge => <EdgeRow key={`${edge.game.matchup}-${edge.side}`} edge={edge} league={league} date={date} />)}</ul>
        </details>
      )}
      <p className="mt-2 text-[11px] leading-snug text-[#a9bec0]">Edge is the game model&apos;s win chance against the sportsbook price, an estimate, not a guarantee. Spread and total edges need a priced line, which ESPN does not provide.</p>
    </section>
  );
}
