"use client";

import { useState, type FormEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, Plus } from "lucide-react";
import { api } from "@/lib/api";
import { useSlip } from "@/store/slip";
import { trendLeg } from "@/lib/trend-leg.mjs";
import type { TrendCard } from "@/lib/types";

const focus = "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400";

/** Adds a trend to the slip. A live sportsbook price is used as is; a derived line has none, so the user supplies it. */
export function TrendSlipButton({ card, league }: { card: TrendCard; league: "nba" | "wnba" }) {
  const legs = useSlip(state => state.legs);
  const toggle = useSlip(state => state.toggle);
  const games = useQuery({ queryKey: ["games", league], queryFn: () => api.games(league), staleTime: 300_000 });
  const date = games.data?.target_date ?? "";
  const [asking, setAsking] = useState(false);
  const [text, setText] = useState("");
  const [invalid, setInvalid] = useState(false);

  const stat = card.stat === "FG3M" ? "3PM" : card.stat;
  const direction = card.side === "under" ? "Under" : "Over";
  const line = card.live?.line ?? card.line;
  const existing = legs.find(leg => leg.league === league && leg.date === date && leg.player === card.player && leg.stat === card.stat && leg.direction === direction && leg.line === line);
  const label = `${card.player} ${direction} ${line} ${stat}`;

  const add = (price?: number) => {
    const leg = trendLeg({ card, league, date, price });
    if (leg) toggle(leg);
    return leg !== null;
  };
  const press = () => {
    if (existing) return toggle(existing);
    if (!add()) setAsking(open => !open);
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const accepted = add(text.trim() === "" ? undefined : Number(text));
    setInvalid(!accepted);
    if (accepted) { setAsking(false); setText(""); }
  };

  return <div className="relative shrink-0" onKeyDown={event => { if (event.key === "Escape") setAsking(false); }}>
    <button type="button" disabled={!date} aria-pressed={!!existing} aria-expanded={asking} onClick={press}
      aria-label={!date ? "No slate date yet, cannot add to slip" : `${existing ? "Remove" : "Add"} ${label} ${existing ? "from" : "to"} slip`}
      className={`grid h-9 w-9 place-items-center rounded-full border transition-[transform,background-color] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97] disabled:opacity-30 ${focus} ${existing ? "border-teal-400 bg-teal-400 text-bg-root" : "border-teal-400/45 text-teal-400 [@media(hover:hover)]:hover:bg-teal-400/10"}`}>
      {existing ? <Check size={16} aria-hidden="true" /> : <Plus size={16} aria-hidden="true" />}
    </button>
    {asking && !existing && <form onSubmit={submit} className="ql-panel absolute right-0 top-11 z-20 w-60 p-3 shadow-xl">
      <label htmlFor={`price-${label}`} className="ql-data-label block">YOUR PRICE · {direction.toUpperCase()} {line} {stat}</label>
      <p className="mt-1 text-[11px] leading-snug text-text-sec">No sportsbook price is posted for this line, so enter the one you see.</p>
      <div className="mt-2 flex gap-2">
        <input id={`price-${label}`} autoFocus inputMode="numeric" value={text} onChange={event => { setText(event.target.value); setInvalid(false); }} placeholder="-110"
          aria-invalid={invalid} className={`min-w-0 flex-1 rounded border border-ink/15 bg-transparent px-2 py-1.5 text-sm tabular-nums ${focus}`} />
        <button type="submit" className={`rounded bg-teal-400 px-3 text-sm font-bold text-bg-root active:scale-[0.97] ${focus}`}>Add</button>
      </div>
      {invalid && <p role="alert" className="mt-1.5 text-[11px] text-red-300">Enter American odds such as -110 or +120.</p>}
    </form>}
  </div>;
}
