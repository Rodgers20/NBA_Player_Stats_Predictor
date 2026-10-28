"use client";

import { useId, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { Prop } from "@/lib/types";
import { bestBookKeys, edgeTier, formatOdds, hitChips, type Tone } from "@/lib/prop-research.mjs";

const TONE: Record<Tone, string> = {
  good: "border-emerald-400/40 text-emerald-300",
  bad: "border-red-400/40 text-red-300",
  neutral: "border-ink/10 text-text-sec",
};
const EDGE_TIER = ["border-ink/10 text-text-sec", "border-teal-400/30 text-teal-400/80", "border-teal-400/50 text-teal-400", "border-teal-400 bg-teal-400/10 text-teal-400"];
const GRADE_TONE: Record<string, Tone> = { A: "good", B: "good", C: "neutral", D: "bad", F: "bad" };
const chip = "rounded border px-1.5 py-0.5 text-[10px] font-bold tabular-nums";

/** Small research chips in the card's existing chip language; renders nothing when the row carries no research fields. */
export function PropResearchChips({ prop }: { prop: Prop }) {
  const tier = edgeTier(prop.edge_pct);
  const windows = hitChips(prop.hit_windows);
  const grade = prop.matchup_grade && prop.def_rank != null ? prop.matchup_grade : null;
  const splits = (prop.teammate_splits ?? []).slice(0, 2);
  return <>
    {prop.stars ? <span className="text-[11px] tracking-tight text-lock-gold" role="img" aria-label={`${prop.stars} of 5 stars`} title="Star rating from the model's edge over the market">{"★".repeat(prop.stars)}<span className="text-ink/20">{"★".repeat(Math.max(0, 5 - prop.stars))}</span></span> : null}
    {prop.high_edge && <span className={`${chip} border-teal-400 bg-teal-400/10 text-teal-400`}>HIGH EDGE</span>}
    {tier != null && <span className={`${chip} ${EDGE_TIER[tier]}`} title="Model probability minus the market's implied probability">EDGE {prop.edge_pct! >= 0 ? "+" : ""}{prop.edge_pct!.toFixed(1)}%</span>}
    {grade && <span className={`${chip} ${TONE[GRADE_TONE[grade] ?? "neutral"]}`} title={`Opponent defense vs this stat, 1 = softest. Matchup grade ${grade}`}>vs {prop.opponent}: #{prop.def_rank}{prop.def_rank_of ? ` of ${prop.def_rank_of}` : ""}</span>}
    {windows.map(w => <span key={w.key} className={`${chip} ${TONE[w.tone]}`} title={`${w.label}: ${w.hits}/${w.total} cleared the posted line`}>{w.label} {w.hits}/{w.total}</span>)}
    {splits.map(s => <span key={s.teammate} title={`${s.games_without} games without ${s.teammate}`}>Without {s.teammate} ({s.status}): <strong className="tabular-nums text-text-pri">{s.avg_without.toFixed(1)}</strong> vs {s.avg_with.toFixed(1)}{s.hit_rate_without != null && <> · <strong className="tabular-nums text-text-pri">{Math.round(s.hit_rate_without)}%</strong> hit</>}</span>)}
  </>;
}

/** Expandable per-book line table with the best over/under outlined and a Best/Consensus/Open summary. */
export function PropBooks({ prop, className = "" }: { prop: Prop; className?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const books = prop.books ?? [];
  if (!books.length) return null;
  const overBest = bestBookKeys(books, "over");
  const underBest = bestBookKeys(books, "under");
  const outline = "rounded px-1 font-bold text-teal-400 outline outline-1 -outline-offset-1 outline-teal-400";
  const summary = [
    prop.best_over && `Best over ${formatOdds(prop.best_over.price)} ${prop.best_over.book}`,
    prop.best_under && `Best under ${formatOdds(prop.best_under.price)} ${prop.best_under.book}`,
    prop.consensus && `Consensus ${prop.consensus.line}`,
    prop.open_line != null && `Open ${prop.open_line}`,
  ].filter(Boolean);
  return <div className={className}>
    <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(v => !v)} className="inline-flex items-center gap-1 text-[11px] font-semibold text-teal-400 transition-transform active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
      Books ({books.length}) <ChevronDown size={12} className={`transition-transform duration-150 ease-out ${open ? "rotate-180" : ""}`} />
    </button>
    {open && <div id={id} className="mt-2 max-w-md text-[11px] text-text-sec">
      {summary.length > 0 && <p className="mb-1.5 flex flex-wrap gap-x-3">{summary.map(s => <span key={String(s)}>{s}</span>)}</p>}
      <table className="w-full border-collapse">
        <thead><tr className="ql-data-label text-[10px]"><th className="px-2 py-1 text-left font-normal">BOOK</th><th className="px-2 py-1 text-right font-normal">LINE</th><th className="px-2 py-1 text-right font-normal">OVER</th><th className="px-2 py-1 text-right font-normal">UNDER</th></tr></thead>
        <tbody>{books.map(b => <tr key={b.book} className="border-t border-ink/[0.06]">
          <td className="px-2 py-1 text-text-pri">{b.book}</td>
          <td className="px-2 py-1 text-right tabular-nums">{b.line}</td>
          <td className="px-2 py-1 text-right tabular-nums"><span className={overBest.has(b.book) ? outline : ""}>{formatOdds(b.over_price) ?? "—"}</span></td>
          <td className="px-2 py-1 text-right tabular-nums"><span className={underBest.has(b.book) ? outline : ""}>{formatOdds(b.under_price) ?? "—"}</span></td>
        </tr>)}</tbody>
      </table>
    </div>}
  </div>;
}
