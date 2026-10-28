"use client";

import { useState } from "react";
import { Check, Minus, Plus } from "lucide-react";
import { useSlip } from "@/store/slip";
import { decimalOdds, formatOdds } from "@/lib/bet-slip.mjs";
import { hitSummary } from "@/lib/hit-rate.mjs";
import { ASSUMED_PRICE } from "@/lib/team-legs.mjs";
import type { ChartRecord } from "@/lib/types";

interface Props {
  league: "nba" | "wnba";
  player: string;
  stat: string;
  statLabel: string;
  /** Every market the player can be measured on, as [value, label]; picking one here is the same as picking it in the page's stat tabs. */
  stats: readonly (readonly [string, string])[];
  onStatChange: (stat: string) => void;
  games: ChartRecord[];
  defaultLine: number;
  date: string;
  matchup: string;
  accent: string;
}

const STEP = 0.5;
const BREAK_EVEN = (100 / 210) * 100;
const press = "transition-[transform,background-color,border-color,color] duration-150 ease-out active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400";
const one = (value: number) => (Number.isInteger(value) ? String(value) : value.toFixed(1));

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Hit-rate view of one market: pick a line, see how often the player cleared it, add a side to the slip. */
export function PlayerPropLines({ league, player, stat, statLabel, stats: markets, onStatChange, games, defaultLine, date, matchup, accent }: Props) {
  const [line, setLine] = useState(defaultLine);
  const [price, setPrice] = useState("");
  const { legs, toggle } = useSlip();

  const valid = games.filter((game): game is ChartRecord & { value: number } => game.value != null);
  const values = valid.map(game => game.value);
  const over = hitSummary(values, line, "Over");
  const overs = over.hits;
  const unders = hitSummary(values, line, "Under").hits;
  const rate = over.rate;
  const lastFive = valid.slice(-5);
  const yMax = Math.max(line, ...values, 1) * 1.12;
  const quotedPrice = Number(price);
  const priceOk = price.trim() !== "" && decimalOdds(quotedPrice) !== null;
  // A blank price falls back to the standard -110 (flagged on the slip); only a price that was typed wrongly blocks adding.
  const canAdd = priceOk || price.trim() === "";

  const inSlip = (direction: "Over" | "Under") => legs.some(leg => leg.league === league && leg.date === date
    && leg.player === player && leg.stat === stat && leg.line === line && leg.direction === direction);
  function addSide(direction: "Over" | "Under") {
    if (!canAdd) return;
    toggle({ league, date, player, stat, direction, line, price: priceOk ? quotedPrice : ASSUMED_PRICE, assumedPrice: !priceOk,
      probability: null, matchup, addedAt: new Date().toISOString() });
  }
  const nudge = (delta: number) => setLine(Math.max(0, Math.round((line + delta) * 2) / 2));

  const stats: [string, string][] = values.length ? [
    ["Average", one(values.reduce((sum, value) => sum + value, 0) / values.length)],
    ["Median", one(median(values))],
    ["Last 5 avg", one(lastFive.reduce((sum, game) => sum + game.value, 0) / lastFive.length)],
    ["High", one(Math.max(...values))],
    ["Low", one(Math.min(...values))],
  ] : [];

  return <section className="ql-panel p-4 sm:p-5" aria-labelledby="prop-lines-title">
    <p className="ql-kicker">PROP LINES</p>
    <h2 id="prop-lines-title" className="ql-section-title mt-1">{statLabel} · set a line</h2>
    <div className="mt-3 flex flex-wrap gap-1.5" role="group" aria-label="Prop market">
      {markets.map(([value, label]) => <button key={value} type="button" aria-pressed={value === stat} onClick={() => onStatChange(value)}
        className={`rounded border px-2.5 py-1 text-xs font-bold ${press} ${value === stat ? "border-teal-400 bg-teal-400 text-bg-root" : "border-ink/15 text-text-sec hover:border-teal-400/50 hover:text-text-pri"}`}>{label}</button>)}
    </div>
    <p className="mb-4 mt-1 text-xs text-text-sec">Set a line to see how often {player} cleared it over the games selected above, then add Over or Under to your bet slip. Bars are games, oldest first.</p>

    {valid.length === 0 ? <p className="py-10 text-center text-sm text-text-sec">No completed games to measure a line against.</p>
    : <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-3">
        <div className="border border-ink/10 bg-ink/[.025] p-4">
          <p className="ql-data-label">LINE · {statLabel}</p>
          <div className="mt-2 flex items-center gap-2">
            <button type="button" aria-label="Lower line" onClick={() => nudge(-STEP)} className={`ql-control flex w-10 items-center justify-center hover:border-teal-400/50 ${press}`}><Minus size={16} /></button>
            <input aria-label={`${statLabel} line`} type="number" inputMode="decimal" step={STEP} min={0} value={line}
              onChange={event => { const next = Number(event.target.value); if (Number.isFinite(next) && next >= 0) setLine(next); }}
              className="ql-control min-w-0 flex-1 px-2 text-center text-xl font-bold tabular-nums outline-none focus:border-teal-400" />
            <button type="button" aria-label="Raise line" onClick={() => nudge(STEP)} className={`ql-control flex w-10 items-center justify-center hover:border-teal-400/50 ${press}`}><Plus size={16} /></button>
          </div>
          <p className="mt-4 flex items-baseline gap-2">
            <strong className="text-4xl font-black tabular-nums" style={{ color: accent }}>{rate == null ? "—" : `${rate}%`}</strong>
            <span className="text-xs text-text-sec">over in {overs} of {over.hits + over.misses} games{over.pushes ? ` · ${over.pushes} push` : ""}</span>
          </p>
          <div className="mt-2 flex items-center gap-1.5">
            <span className="ql-data-label mr-1">Last {lastFive.length}</span>
            {lastFive.map((game, index) => {
              const label = game.value > line ? "O" : game.value < line ? "U" : "P";
              const tone = label === "O" ? "bg-emerald-500/80 text-[#06130f]" : label === "U" ? "bg-red-400/80 text-[#1a0808]" : "bg-ink/20 text-text-pri";
              return <span key={`${game.date}-${index}`} title={`${game.date} · ${game.value}`} className={`flex h-6 w-6 items-center justify-center rounded text-[11px] font-black ${tone}`}>{label}</span>;
            })}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 border-t border-ink/10 pt-4">
            {(["Over", "Under"] as const).map(side => {
              const added = inSlip(side);
              return <button key={side} type="button" onClick={() => addSide(side)} disabled={!canAdd} aria-pressed={added}
                title={canAdd ? undefined : "That price isn't valid. Use American odds, e.g. -110"}
                className={`ql-control flex flex-col items-center justify-center gap-0.5 px-2 py-2 text-sm font-bold hover:border-teal-400/50 disabled:cursor-not-allowed disabled:opacity-40 ${press} ${added ? "!border-teal-400 !bg-teal-400 !text-bg-root" : ""}`}>
                <span className="flex items-center gap-1">{added ? <Check size={15} /> : <Plus size={15} />}{added ? "In slip" : side}</span>
                <span className="text-[11px] font-semibold tabular-nums opacity-80">{side} {one(line)}</span>
              </button>;
            })}
          </div>
          <label className="mt-3 flex items-center justify-between gap-3 text-xs text-text-sec">
            <span>Your price (American)</span>
            <input value={price} placeholder="-115" onChange={event => setPrice(event.target.value)} inputMode="numeric" aria-invalid={price !== "" && !priceOk}
              className="ql-control w-20 px-2 text-center text-sm font-bold tabular-nums outline-none focus:border-teal-400 aria-[invalid=true]:border-red-400" />
          </label>
          <p className="mt-1 text-[11px] text-text-sec">{priceOk ? `The slip will use ${formatOdds(quotedPrice)}, the price you entered.` : canAdd ? `No price entered: the slip uses ${formatOdds(ASSUMED_PRICE)}, marked as assumed. Enter your book's price to replace it.` : "Enter American odds such as -115 or +120."}</p>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {stats.map(([label, value]) => <div key={label} className="border border-ink/10 bg-ink/[.025] p-3">
            <p className="ql-data-label">{label}</p><p className="mt-1 text-xl font-bold tabular-nums text-text-pri">{value}</p>
          </div>)}
        </div>
      </div>

      <div className="min-w-0 overflow-x-auto">
        <div className="relative flex h-[300px] min-w-[420px] items-end gap-2 border-b border-ink/15 pb-0 pl-1 pr-14 pt-6" role="img"
          aria-label={`${statLabel} per game against a line of ${one(line)}: ${overs} over, ${unders} under`}>
          <div className="pointer-events-none absolute inset-x-0 border-t border-dashed border-ink/70" style={{ bottom: `${(line / yMax) * 100}%` }}>
            <span className="absolute -top-2.5 right-0 text-xs font-bold tabular-nums text-text-pri">{one(line)}</span>
          </div>
          {valid.map((game, index) => {
            const over = game.value > line;
            const under = game.value < line;
            return <div key={`${game.date}-${index}`} className="relative flex h-full min-w-[22px] flex-1 flex-col justify-end">
              <span className="mb-1 text-center text-[11px] font-bold tabular-nums text-text-sec">{one(game.value)}</span>
              <div className="rounded-t-[3px] transition-colors duration-150 ease-out"
                style={{ height: `${(game.value / yMax) * 100}%`, background: over ? "rgba(34,197,94,.78)" : under ? "rgba(239,68,68,.72)" : "color-mix(in srgb, var(--color-ink) 28%, transparent)" }} />
            </div>;
          })}
        </div>
        <div className="flex min-w-[420px] gap-2 pl-1 pr-14 pt-1.5">
          {valid.map((game, index) => <span key={`${game.date}-${index}`} title={game.date} className="min-w-[22px] flex-1 truncate text-center text-[10px] text-text-sec">{game.opponent.replace(/^(vs\.?\s*)/i, "")}</span>)}
        </div>
      </div>
    </div>}

    <p className="mt-4 border-t border-ink/10 pt-3 text-xs text-text-sec">This is what already happened, not a projection. At −110 a bet needs to win {BREAK_EVEN.toFixed(1)}% to break even, and a handful of games swings a hit rate a long way. A game that lands exactly on the line counts as neither Over nor Under.</p>
  </section>;
}
