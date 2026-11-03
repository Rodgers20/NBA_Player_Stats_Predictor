import { Panel } from "@/components/games/shared/Panel";
import { CONFIDENCE_LABEL, american, lineLabel, marketOf, pct } from "@/lib/nhl.mjs";
import type { NhlTopProp } from "@/lib/nhl-types";
import { Headshot } from "./parts";

/** The game's best model-probability props. `onOpen` opens the full player analysis. */
export function NhlTopProps({ props, onOpen }: { props: NhlTopProp[]; onOpen: (id: number) => void }) {
  if (!props.length) return null;
  return (
    <Panel title="Top prop picks" note="Model probabilities with fair odds, no sportsbook line. Compare fair odds with your book: a positive edge needs the book to pay better than fair.">
      <ul className="divide-y divide-ink/[0.07]">
        {props.map(p => (
          <li key={`${p.player_id}-${p.market}-${p.line}`}>
            <button type="button" onClick={() => onOpen(p.player_id)} className="grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 py-2.5 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
              <Headshot src={p.headshot} name={p.name} size={40} />
              <span className="min-w-0">
                <span className="block truncate font-bold text-text-pri">{p.name} <span className="font-mono text-[10px] font-normal text-text-sec">{p.team} · {p.pos}</span></span>
                <span className="block truncate text-xs text-text-sec">{marketOf(p.market)?.label}{p.market === "goal" ? "" : ` ${lineLabel(p.market, p.line)}`} · {CONFIDENCE_LABEL[p.confidence]?.toLowerCase()}</span>
              </span>
              <span className="text-right"><span className="block font-mono text-base font-bold tabular-nums text-teal-300">{pct(p.p)}</span><span className="block font-mono text-[11px] text-text-sec">fair {american(p.fair_odds)}</span></span>
            </button>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
