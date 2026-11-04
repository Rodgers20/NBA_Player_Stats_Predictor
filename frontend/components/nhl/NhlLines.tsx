import { Panel } from "@/components/games/shared/Panel";
import { american, pct, signed } from "@/lib/nhl.mjs";
import type { NhlGame } from "@/lib/nhl-types";

const plus = (v: number | null) => (v == null ? "—" : american(v) ?? "—");
/** Positive expected value at the book's price, as "+4.1%". */
const evText = (ev: number | null) => (ev == null ? null : `${ev >= 0 ? "+" : ""}${(ev * 100).toFixed(1)}%`);

/** The side with positive model edge against ESPN's price, if any. */
function valueSide(g: NhlGame) {
  const l = g.lines;
  if (!l || l.ev_home == null || l.ev_away == null) return null;
  const home = l.ev_home >= l.ev_away;
  const ev = home ? l.ev_home : l.ev_away;
  return ev > 0.02 ? { abbr: home ? g.home.abbr : g.away.abbr, ev } : null;
}

/** One line of sportsbook prices under a slate card: moneyline, total, puck line. Hidden until ESPN posts a line. */
export function LinesStrip({ game }: { game: NhlGame }) {
  const l = game.lines;
  if (!l) return null;
  const v = valueSide(game);
  const puck = l.puck_home == null ? null : l.puck_home;
  return (
    <p className="relative mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-ink/[0.07] pt-2 font-mono text-[10px] text-text-sec" aria-label="Sportsbook line">
      <span>ML <b className="text-text-pri">{game.away.abbr} {plus(l.away_ml)}</b> · <b className="text-text-pri">{game.home.abbr} {plus(l.home_ml)}</b></span>
      {l.total != null && <span>O/U <b className="text-text-pri">{l.total}</b></span>}
      {puck != null && <span>{game.home.abbr} <b className="text-text-pri">{signed(puck)}</b></span>}
      {v && <span className="font-bold text-teal-300" title="Model win chance against the no-vig price">Value {v.abbr} {evText(v.ev)}</span>}
    </p>
  );
}

/** Game page panel: both sides' prices next to the model, with the line's movement since it was first seen. */
export function LinesPanel({ game }: { game: NhlGame }) {
  const l = game.lines;
  if (!l) return null;
  const p = game.pick;
  const moved = (now: number | null, was: number | null | undefined) => (now != null && was != null && now !== was ? ` (opened ${was > 0 ? "+" : ""}${was})` : "");
  const rows = [
    { team: game.away, ml: l.away_ml, ev: l.ev_away, model: p?.p_away, market: l.market_p_home == null ? null : 1 - l.market_p_home, spread: l.puck_home == null ? null : -l.puck_home, spreadOdds: l.away_spread_odds, open: l.open?.away_ml },
    { team: game.home, ml: l.home_ml, ev: l.ev_home, model: p?.p_home, market: l.market_p_home, spread: l.puck_home, spreadOdds: l.home_spread_odds, open: l.open?.home_ml },
  ];
  return (
    <Panel title="Sportsbook line" note={`${l.book ?? "Sportsbook"} via ESPN, moneyline, puck line and total only. ESPN has no player prop lines, so props stay model-only. Value compares the model's win chance with the price after removing the book's margin.`}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] text-sm">
          <thead><tr className="ql-data-label text-left"><th className="py-1 font-normal">Team</th><th className="py-1 text-right font-normal">ML</th><th className="py-1 text-right font-normal">Puck line</th><th className="py-1 text-right font-normal">Market</th><th className="py-1 text-right font-normal">Model</th><th className="py-1 text-right font-normal">Value</th></tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.team.abbr} className="border-t border-ink/[0.07] font-mono tabular-nums">
                <td className="py-2 font-display font-bold text-text-pri">{r.team.abbr}</td>
                <td className="py-2 text-right text-text-pri">{plus(r.ml)}<span className="text-[10px] text-text-sec">{moved(r.ml, r.open)}</span></td>
                <td className="py-2 text-right text-text-pri">{r.spread == null ? "—" : `${signed(r.spread)} (${plus(r.spreadOdds)})`}</td>
                <td className="py-2 text-right text-text-sec">{pct(r.market)}</td>
                <td className="py-2 text-right text-text-sec">{pct(r.model)}</td>
                <td className={`py-2 text-right font-bold ${r.ev != null && r.ev > 0.02 ? "text-teal-300" : "text-text-sec"}`}>{evText(r.ev) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {l.total != null && <p className="mt-3 font-mono text-xs text-text-sec">Total <b className="text-text-pri">{l.total}</b> · over {plus(l.over_odds)} / under {plus(l.under_odds)}{p ? <> · model projects <b className="text-text-pri">{p.proj_total.toFixed(1)}</b></> : null}</p>}
    </Panel>
  );
}
