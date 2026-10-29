"use client";

import { useState } from "react";
import Image from "next/image";
import { PlayerName, focusFromProp } from "@/components/analysis/PlayerName";
import { Check, Plus } from "lucide-react";
import { useSlip } from "@/store/slip";
import { decimalOdds, formatOdds } from "@/lib/bet-slip.mjs";
import { rowDashes, windowCell } from "@/lib/prop-rows.mjs";
import { HitDashes } from "@/components/ui/HitDashes";
import type { Game, Prop } from "@/lib/types";

const tipTime = (game?: Game) => {
  const date = game ? new Date(game.start_time || game.game_time) : null;
  return date && !Number.isNaN(date.getTime())
    ? `${date.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" })} ET`
    : "";
};

interface Props {
  prop: Prop;
  rank: number;
  league: "nba" | "wnba";
  date: string;
  game?: Game;
}

/** One popular play: who, which side of which line, how often it cleared lately, and the price to take it at. */
export function PopularCard({ prop, rank, league, date, game }: Props) {
  const [photoFailed, setPhotoFailed] = useState(false);
  const { legs, toggle } = useSlip();
  const side = /^under$/i.test(prop.direction) ? "Under" : "Over";
  const last10 = windowCell(prop, "L10");
  const canAdd = prop.line != null && prop.price != null && decimalOdds(prop.price) !== null;
  const inSlip = legs.some(leg => leg.league === league && leg.date === date && leg.player === prop.player
    && leg.stat === prop.stat && leg.line === prop.line && leg.direction.toLowerCase() === side.toLowerCase());
  const add = () => {
    if (!canAdd || prop.line == null || prop.price == null) return;
    toggle({ league, date, player: prop.player, stat: prop.stat, direction: side, line: prop.line, price: prop.price,
      probability: prop.model_prob ?? null, matchup: prop.game_matchup, team: prop.team, addedAt: new Date().toISOString() });
  };
  const initial = prop.player.trim().split(/\s+/).at(-1)?.[0]?.toUpperCase() || "?";
  const tip = tipTime(game);
  const ev = prop.ev == null ? null : `${prop.ev >= 0 ? "+" : ""}${(prop.ev * 100).toFixed(1)}% EV`;

  return (
    <article className="grid grid-cols-[auto_auto_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-3 border-b border-ink/[0.07] px-4 py-4 last:border-b-0 sm:px-5 md:grid-cols-[auto_auto_minmax(0,1.5fr)_minmax(0,1.1fr)_auto]">
      <span className="w-5 text-center text-xs font-bold tabular-nums text-text-sec" aria-label={`Rank ${rank}`}>{rank}</span>
      <div className="flex h-11 w-11 shrink-0 items-center justify-center overflow-hidden rounded-md border border-ink/10 bg-teal-400/10 text-sm font-bold text-teal-400" aria-hidden="true">
        {prop.headshot_url && !photoFailed
          ? <Image src={prop.headshot_url} alt="" width={44} height={44} unoptimized onError={() => setPhotoFailed(true)} className="h-11 w-11 object-cover object-top" />
          : initial}
      </div>
      <div className="min-w-0">
        <p className="text-xs leading-snug text-text-sec">{prop.game_matchup}{tip && ` · ${tip}`}</p>
        <p className="break-words text-sm font-bold leading-snug text-text-pri">
          <PlayerName name={prop.player} league={league} focus={focusFromProp(prop)} />
        </p>
        <p className="text-sm font-bold leading-snug text-teal-400">{side} {prop.line} {prop.stat_label || prop.stat}</p>
      </div>
      <div className="col-span-2 col-start-3 min-w-0 md:col-span-1 md:col-start-auto">
        {last10 && <>
          <p className="tabular-nums"><span className="text-xl font-extrabold text-text-pri">{last10.pct}%</span> <span className="text-xs text-text-sec">{last10.hits} of {last10.total} last games</span></p>
          <HitDashes dashes={rowDashes(prop)} wide />
        </>}
      </div>
      <div className="col-start-4 row-start-1 flex items-center gap-2 md:col-start-auto md:row-start-auto">
        <div className="text-right">
          <p className="inline-block rounded-md border border-ink/10 bg-ink/[0.06] px-2 py-1 font-mono text-sm font-semibold tabular-nums text-text-pri">{formatOdds(prop.price)}</p>
          <p className="mt-1 text-[10px] uppercase leading-tight tracking-wide text-text-sec">{prop.book ?? "Sportsbook"}</p>
          {ev && <p className="text-[10px] uppercase leading-tight tracking-wide text-text-sec">{ev}</p>}
        </div>
        <button type="button" onClick={add} disabled={!canAdd} aria-pressed={inSlip}
          aria-label={inSlip ? `Remove ${prop.player} ${prop.stat} from slip` : `Add ${prop.player} ${prop.stat} to slip`}
          className="grid h-9 w-9 place-items-center rounded-lg border border-teal-400/40 text-teal-400 transition-transform duration-150 active:scale-95 disabled:opacity-30 aria-pressed:border-teal-400 aria-pressed:bg-teal-400 aria-pressed:text-bg-root focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
          {inSlip ? <Check size={17} /> : <Plus size={17} />}
        </button>
      </div>
    </article>
  );
}
