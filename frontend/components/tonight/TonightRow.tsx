"use client";

import { useState } from "react";
import Image from "next/image";
import { Check, Plus } from "lucide-react";
import { useSlip } from "@/store/slip";
import { decimalOdds, formatOdds } from "@/lib/bet-slip.mjs";
import { playReason, quoteAge } from "@/lib/tonight.mjs";
import type { Prop } from "@/lib/types";
import { PlayerName, focusFromProp } from "@/components/analysis/PlayerName";

interface Props {
  prop: Prop;
  league: "nba" | "wnba";
  date: string;
}

const press = "transition-[transform,background-color,border-color,color] duration-150 ease-out active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400";

/** One priced play on one line: the side and price, one number, one reason. Chips, books and history live in the player sheet. */
export function TonightRow({ prop, league, date }: Props) {
  const [photoFailed, setPhotoFailed] = useState(false);
  const { legs, toggle } = useSlip();
  const side = /^under$/i.test(prop.direction) ? "Under" : "Over";
  const age = quoteAge(prop.quote_updated_at);
  const canAdd = prop.line != null && prop.price != null && decimalOdds(prop.price) !== null;
  const inSlip = legs.some(leg => leg.league === league && leg.date === date && leg.player === prop.player
    && leg.stat === prop.stat && leg.line === prop.line && leg.direction.toLowerCase() === side.toLowerCase());
  function add() {
    if (!canAdd || prop.line == null || prop.price == null) return;
    toggle({ league, date, player: prop.player, stat: prop.stat, direction: side, line: prop.line, price: prop.price,
      probability: prop.model_prob ?? null, matchup: prop.game_matchup, team: prop.team, addedAt: new Date().toISOString() });
  }
  const ev = prop.ev == null ? "—" : `${prop.ev >= 0 ? "+" : ""}${(prop.ev * 100).toFixed(1)}%`;
  const model = prop.model_prob == null ? null : `model ${Math.round(prop.model_prob * 100)}%`;
  const reason = playReason(prop);
  const against = reason?.startsWith("Against");
  const initial = prop.player.trim().split(/\s+/).at(-1)?.[0]?.toUpperCase() || "?";

  return <article className="grid grid-cols-[auto_minmax(0,1fr)_auto_auto] items-center gap-x-3 gap-y-2 border-b border-ink/[0.07] px-4 py-3 last:border-b-0 sm:px-5">
    <PlayerName name={prop.player} league={league} focus={focusFromProp(prop)}
      className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-md border border-ink/10 bg-teal-400/10 text-sm font-bold text-teal-400">
      {prop.headshot_url && !photoFailed
        ? <Image src={prop.headshot_url} alt="" width={40} height={40} unoptimized onError={() => setPhotoFailed(true)} className="h-10 w-10 object-cover object-top" />
        : initial}
    </PlayerName>
    <div className="min-w-0">
      <p className="truncate text-sm font-bold text-text-pri">
        <PlayerName name={prop.player} league={league} focus={focusFromProp(prop)} />
        <span className="text-teal-400"> · {side} {prop.line} {prop.stat_label || prop.stat}</span>
      </p>
      <p className="truncate text-xs tabular-nums text-text-sec">
        {prop.game_matchup || `${prop.team} vs ${prop.opponent}`} · <strong className="text-text-pri">{formatOdds(prop.price)}</strong> {prop.book ?? ""}
        {age?.stale && <span className="text-amber-300"> · priced {age.label}</span>}
      </p>
    </div>
    <div className="text-right">
      <p className="font-mono text-lg font-bold tabular-nums text-teal-400">{ev}</p>
      {model && <p className="text-[11px] tabular-nums text-text-sec">{model}</p>}
    </div>
    <button type="button" onClick={add} disabled={!canAdd} aria-pressed={inSlip}
      aria-label={inSlip ? `Remove ${prop.player} ${side} ${prop.line} from slip` : `Add ${prop.player} ${side} ${prop.line} to slip`}
      className={`ql-control flex min-h-10 items-center justify-center gap-1 px-3 text-sm font-bold hover:border-teal-400/50 disabled:cursor-not-allowed disabled:opacity-40 ${press} ${inSlip ? "!border-teal-400 !bg-teal-400 !text-bg-root" : "text-teal-400"}`}>
      {inSlip ? <Check size={16} /> : <Plus size={16} />}<span className="hidden sm:inline">{inSlip ? "In slip" : "Add"}</span>
    </button>
    {reason && <p className={`col-span-4 border-t border-dashed border-ink/[0.08] pt-2 text-xs leading-snug ${against ? "text-amber-200" : "text-text-sec"}`}>{reason}</p>}
  </article>;
}
