import Link from "next/link";
import { CONFIDENCE_LABEL, MARKETS, american, lineLabel, num, pct } from "@/lib/nhl.mjs";
import type { NhlPlayerDetail } from "@/lib/nhl-types";
import { AbbrLogo, FairOddsNote } from "./parts";

const day = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
const rank = (n: number | null) => { if (n == null) return "—"; const t = ["th", "st", "nd", "rd"], v = n % 100; return `${n}${t[(v - 20) % 10] ?? t[v] ?? t[0]}`; };

/** The player's next game: model probabilities and fair odds per market and line, plus the opponent's context. */
export function NhlNextGame({ d }: { d: NhlPlayerDetail }) {
  const n = d.next;
  if (!n) return <p role="status" className="ql-panel p-5 text-sm text-text-sec">{d.player.team} has no scheduled game in the data window, so there are no projections yet.</p>;
  const ctx = n.opp_context;
  return (
    <section aria-label="Next game" className="space-y-3">
      <div className="ql-panel p-4">
        <p className="ql-kicker">Next game</p>
        <h3 className="mt-1 flex flex-wrap items-center gap-2 font-display text-xl font-bold">
          <AbbrLogo abbr={n.opp} size={26} />{n.home ? "vs" : "at"} {n.opp}<span className="font-mono text-xs font-medium text-text-sec">{day(n.date)}</span>
          <Link href={`/nhl/game?id=${n.game_id}`} className="ml-auto text-xs font-semibold text-teal-300 underline">Game page</Link>
        </h3>
        {n.lineup === "projected" && <p className="mt-1 text-xs text-text-sec">Projected lineup: confirm he is dressing before betting.</p>}
        {ctx && (
          <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div><dt className="ql-data-label">{n.opp} shots allowed</dt><dd className="font-mono text-sm font-bold">{num(ctx.sa_pg)} <span className="text-[11px] font-normal text-text-sec">{rank(ctx.rank_sa)} most</span></dd></div>
            <div><dt className="ql-data-label">{n.opp} goals allowed</dt><dd className="font-mono text-sm font-bold">{num(ctx.ga_pg, 2)} <span className="text-[11px] font-normal text-text-sec">{rank(ctx.rank_ga)} most</span></dd></div>
            <div><dt className="ql-data-label">Opposing goalie</dt><dd className="font-mono text-sm font-bold">{ctx.opp_goalie ?? "—"}</dd></div>
            <div><dt className="ql-data-label">Goalie save %</dt><dd className="font-mono text-sm font-bold">{ctx.goalie_sv_pct != null ? ctx.goalie_sv_pct.toFixed(3).replace(/^0/, "") : "—"}</dd></div>
          </dl>
        )}
      </div>
      <FairOddsNote />
      <div className="ql-panel overflow-x-auto">
        <table className="w-full min-w-[520px] text-sm">
          <thead><tr className="border-b border-border-card text-left">
            {["Market", "Projection", "Over", "Prob", "Fair odds", "L10", "L20", "Season"].map((h, i) => <th key={h} scope="col" className={`ql-data-label px-3 py-3 ${i > 0 ? "text-right" : ""}`}>{h}</th>)}
          </tr></thead>
          <tbody>
            {MARKETS.flatMap(m => {
              const mk = n.markets[m.key];
              return (mk?.lines ?? []).map((l, i) => (
                <tr key={`${m.key}-${l.line}`} className="border-b border-ink/[0.06] last:border-0">
                  <td className="px-3 py-2 font-bold text-text-pri">{i === 0 ? m.label : ""}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs tabular-nums" title={mk ? CONFIDENCE_LABEL[mk.confidence] : undefined}>{i === 0 && mk ? num(mk.projection, 2) : ""}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs">{l.line} ({lineLabel(m.key, l.line)})</td>
                  <td className="px-3 py-2 text-right font-mono text-xs font-bold tabular-nums text-teal-300">{pct(l.p_over)}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs tabular-nums">{american(l.fair_over)}</td>
                  {[l.hit_l10, l.hit_l20, l.hit_season].map((h, j) => <td key={j} className="px-3 py-2 text-right font-mono text-xs tabular-nums text-text-soft">{pct(h)}</td>)}
                </tr>
              ));
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
