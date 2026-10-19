"use client";

import { useState } from "react";
import Image from "next/image";
import { Check, Plus } from "lucide-react";
import { useSlip } from "@/store/slip";
import { decimalOdds, formatOdds } from "@/lib/bet-slip.mjs";
import { breakEven, quoteAge, recordFor } from "@/lib/tonight.mjs";
import type { Prop } from "@/lib/types";

interface Props {
  prop: Prop;
  league: "nba" | "wnba";
  date: string;
  record?: Parameters<typeof recordFor>[0];
}

const press = "transition-[transform,background-color,border-color,color] duration-150 ease-out active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8]";

function Figure({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return <div className="min-w-0">
    <p className="ql-data-label text-[10px]">{label}</p>
    <p className="text-lg font-bold tabular-nums text-text-pri">{value}</p>
    {hint && <p className="text-[11px] tabular-nums text-text-sec">{hint}</p>}
  </div>;
}

/** One bettable play: the side, the price and where it came from, and the three numbers behind it. */
export function TonightRow({ prop, league, date, record }: Props) {
  const [photoFailed, setPhotoFailed] = useState(false);
  const { legs, toggle } = useSlip();
  const side = /^under$/i.test(prop.direction) ? "Under" : "Over";
  const age = quoteAge(prop.quote_updated_at);
  const needed = breakEven(prop.price ?? NaN);
  const canAdd = prop.line != null && prop.price != null && decimalOdds(prop.price) !== null;
  const inSlip = legs.some(leg => leg.league === league && leg.date === date && leg.player === prop.player
    && leg.stat === prop.stat && leg.line === prop.line && leg.direction.toLowerCase() === side.toLowerCase());
  function add() {
    if (!canAdd || prop.line == null || prop.price == null) return;
    toggle({ league, date, player: prop.player, stat: prop.stat, direction: side, line: prop.line, price: prop.price,
      probability: prop.model_prob ?? null, matchup: prop.game_matchup, addedAt: new Date().toISOString() });
  }
  const ev = prop.ev == null ? "—" : `${prop.ev >= 0 ? "+" : ""}${(prop.ev * 100).toFixed(1)}%`;
  const model = prop.model_prob == null ? "—" : `${Math.round(prop.model_prob * 100)}%`;
  const sample = prop.hits != null && prop.total ? `${prop.hits}/${prop.total}` : "—";
  const graded = recordFor(record, prop.stat);
  const modelHint = [prop.market_prob != null && `market ${Math.round(prop.market_prob * 100)}%`,
    needed != null && `needs ${needed.toFixed(1)}%`].filter(Boolean).join(" · ") || undefined;
  const initial = prop.player.trim().split(/\s+/).at(-1)?.[0]?.toUpperCase() || "?";

  return <article className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-3 border-b border-white/[0.07] px-4 py-4 last:border-b-0 sm:px-5 md:grid-cols-[auto_minmax(0,1.3fr)_minmax(0,1.4fr)_auto]">
    <div className="flex h-11 w-11 shrink-0 items-center justify-center overflow-hidden rounded-md border border-white/10 bg-[#59e0c8]/10 text-sm font-bold text-[#59e0c8]" aria-hidden="true">
      {prop.headshot_url && !photoFailed
        ? <Image src={prop.headshot_url} alt="" width={44} height={44} unoptimized onError={() => setPhotoFailed(true)} className="h-11 w-11 object-cover object-top" />
        : initial}
    </div>
    <div className="min-w-0">
      <p className="truncate text-sm font-bold text-text-pri">{prop.player}</p>
      <p className="truncate text-xs text-text-sec">{prop.game_matchup || `${prop.team} vs ${prop.opponent}`}</p>
      <p className="mt-1.5 text-base font-extrabold text-[#59e0c8]">{side} {prop.line} {prop.stat_label || prop.stat}</p>
      <p className="text-xs tabular-nums text-text-sec">
        <strong className="text-text-pri">{formatOdds(prop.price)}</strong>
        {" · "}{prop.book ?? "book not recorded"}
        {age && <span className={age.stale ? "text-amber-300" : undefined}> · {age.label}{age.stale ? " — verify the price" : ""}</span>}
      </p>
    </div>
    <button type="button" onClick={add} disabled={!canAdd} aria-pressed={inSlip}
      aria-label={inSlip ? `Remove ${prop.player} ${side} ${prop.line} from slip` : `Add ${prop.player} ${side} ${prop.line} to slip`}
      className={`ql-control flex min-h-11 min-w-[88px] items-center justify-center gap-1.5 px-3 text-sm font-bold hover:border-[#59e0c8]/50 disabled:cursor-not-allowed disabled:opacity-40 md:order-4 ${press} ${inSlip ? "!border-[#59e0c8] !bg-[#59e0c8] !text-[#0d151b]" : "text-[#59e0c8]"}`}>
      {inSlip ? <Check size={16} /> : <Plus size={16} />}{inSlip ? "In slip" : "Add"}
    </button>
    <div className="col-span-3 grid grid-cols-3 gap-3 border-t border-white/[0.06] pt-3 md:order-3 md:col-span-1 md:border-t-0 md:pt-0">
      <Figure label="EDGE (EV)" value={ev} />
      <Figure label="MODEL" value={model} hint={modelHint} />
      <Figure label="LAST GAMES" value={sample} hint="hit this side" />
    </div>
    <p className="col-span-3 text-[11px] leading-snug text-text-sec md:order-5 md:col-span-4">
      {prop.raw_model_prob != null && <>Raw model {Math.round(prop.raw_model_prob * 100)}%, pulled toward the market for a {prop.total}-game sample. </>}
      {graded
        ? <>Graded {prop.stat_label || prop.stat} picks: <strong className="tabular-nums text-text-pri">{graded.pct}%</strong> ({graded.hit}/{graded.total}){graded.early ? " · early sample" : ""}.</>
        : <>No graded {prop.stat_label || prop.stat} picks yet.</>}
    </p>
  </article>;
}
