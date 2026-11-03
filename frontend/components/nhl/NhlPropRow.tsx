import { CONFIDENCE_LABEL, american, lineLabel, num, pct, tipText } from "@/lib/nhl.mjs";
import type { BoardRow } from "@/lib/nhl-props.mjs";
import type { NhlMarketKey } from "@/lib/nhl-types";
import { AbbrLogo, Headshot } from "./parts";

const hitTone = (v: number | null) => (v == null ? "text-text-sec" : v >= 0.7 ? "text-teal-300" : v <= 0.3 ? "text-rose-300" : "text-text-soft");

/** One board row: player and matchup, the model's probability with fair odds, and recent hit rates over the same line. */
export function NhlPropRow({ row, market, onOpen }: { row: BoardRow; market: NhlMarketKey; onOpen: (id: number) => void }) {
  const { player: p, market: m, line: l } = row;
  return (
    <li>
      <button type="button" onClick={() => onOpen(p.player_id)} aria-label={`${p.name}, ${l.line} ${market}, ${pct(l.p_over)}. Open analysis`}
        className="grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-2 px-3 py-3 text-left transition-colors hover:bg-surface-hover/50 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-teal-400 sm:grid-cols-[auto_minmax(0,1fr)_auto_auto_auto] sm:gap-x-5">
        <Headshot src={p.headshot} name={p.name} size={44} />
        <span className="min-w-0">
          <span className="block truncate font-bold text-text-pri">{p.name} <span className="font-mono text-[10px] font-normal text-text-sec">{p.pos}</span></span>
          <span className="flex items-center gap-1.5 truncate font-mono text-[11px] text-text-sec">
            <AbbrLogo abbr={p.team} size={14} />{p.team} {p.home ? "vs" : "@"} {p.opp} · {tipText(p.start).replace(" ET", "")}
          </span>
        </span>
        <span className="text-right sm:order-3">
          <span className="block font-mono text-lg font-bold tabular-nums text-teal-300">{pct(l.p_over)}</span>
          <span className="block font-mono text-[11px] text-text-sec" title="Fair price, no vig">fair {american(l.fair_over)}</span>
        </span>
        <span className="col-span-3 flex items-center gap-x-5 gap-y-1 pl-[56px] font-mono text-[11px] sm:order-2 sm:col-span-1 sm:block sm:pl-0 sm:text-right">
          <span className="block text-text-pri"><b>{market === "goal" ? "Anytime" : `${l.line} (${lineLabel(market, l.line)})`}</b></span>
          <span className="block text-text-sec" title={CONFIDENCE_LABEL[m.confidence]}>proj {num(m.projection, 2)}</span>
        </span>
        <span className="col-span-3 flex gap-4 pl-[56px] font-mono text-[11px] sm:order-4 sm:col-span-1 sm:block sm:pl-0 sm:text-right">
          <span className={`block ${hitTone(l.hit_l10)}`}>L10 {pct(l.hit_l10)}</span>
          <span className={`block ${hitTone(l.hit_l20)}`}>L20 {pct(l.hit_l20)}</span>
        </span>
      </button>
    </li>
  );
}
