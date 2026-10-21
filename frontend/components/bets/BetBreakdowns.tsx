"use client";

import type { BreakdownRow, JournalSummary } from "@/lib/local-journal.mjs";

const money = (value: number) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
const signed = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
const tone = (value: number | null) => value === null || value === 0 ? "" : value > 0 ? "text-green-300" : "text-red-300";

function BreakdownTable({ title, rows }: { title: string; rows: BreakdownRow[] }) {
  return <section className="ql-panel overflow-x-auto" aria-label={`ROI by ${title.toLowerCase()}`}>
    <h3 className="ql-data-label px-4 pt-4">ROI by {title}</h3>
    <table className="mt-2 w-full whitespace-nowrap text-left text-sm">
      <caption className="sr-only">Settled bets grouped by {title.toLowerCase()}</caption>
      <thead><tr className="border-b border-[#2b4248] text-[11px] font-mono uppercase tracking-wider text-[var(--color-text-sec)]">
        {["", "Bets", "W-L-P", "ROI", "Net"].map(label => <th key={label} scope="col" className={`px-4 py-2 font-medium ${label ? "text-right" : ""}`}>{label || title}</th>)}
      </tr></thead>
      <tbody>{rows.map(row => <tr key={row.key} className="border-b border-[#2b4248] last:border-0 hover:bg-[#20363a]">
        <td className="px-4 py-2.5 font-semibold">{row.key}</td>
        <td className="px-4 py-2.5 text-right tabular-nums">{row.bets}</td>
        <td className="px-4 py-2.5 text-right tabular-nums text-[var(--color-text-sec)]">{row.wins}-{row.losses}-{row.pushes}</td>
        <td className={`px-4 py-2.5 text-right tabular-nums font-semibold ${tone(row.roi)}`}>{row.roi === null ? "—" : signed(row.roi)}</td>
        <td className={`px-4 py-2.5 text-right tabular-nums ${tone(row.profit)}`}>{money(row.profit)}</td>
      </tr>)}</tbody>
    </table>
  </section>;
}

const W = 640, H = 160, PAD = 8;

/** Cumulative P&L in settlement order. A zero baseline shows where the line crosses break-even. */
function ProfitCurve({ summary }: { summary: JournalSummary }) {
  const points = summary.curve;
  const values = [0, ...points.map(point => point.pnl)];
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const x = (index: number) => PAD + (index / Math.max(points.length, 1)) * (W - 2 * PAD);
  const y = (value: number) => PAD + (1 - (value - lo) / span) * (H - 2 * PAD);
  const line = ["M" + x(0) + " " + y(0), ...points.map((point, index) => `L${x(index + 1).toFixed(1)} ${y(point.pnl).toFixed(1)}`)].join(" ");
  const last = points.at(-1)?.pnl ?? 0;
  return <section className="ql-panel p-4" aria-label="Cumulative profit and loss">
    <div className="flex flex-wrap items-baseline justify-between gap-2">
      <h3 className="ql-data-label">Cumulative P&amp;L</h3>
      <p className="text-xs text-[var(--color-text-sec)]">Max drawdown <b className="tabular-nums text-[var(--color-text-pri)]">{money(summary.max_drawdown)}</b> · {points.length} settled</p>
    </div>
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 h-40 w-full" role="img" aria-label={`Profit curve ending at ${money(last)}`} preserveAspectRatio="none">
      <line x1={PAD} x2={W - PAD} y1={y(0)} y2={y(0)} stroke="#2b4248" strokeDasharray="4 4" />
      <path d={line} fill="none" stroke={last >= 0 ? "#59e0c8" : "#f87171"} strokeWidth="2" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    </svg>
  </section>;
}

/** ROI breakdowns and the profit curve. The server journal and the local journal return the same summary shape. */
export function BetBreakdowns({ summary }: { summary: JournalSummary }) {
  if (!summary.settled || !summary.curve) return null;
  const tables: [string, BreakdownRow[]][] = [["Stat", summary.by_stat], ["Side", summary.by_side], ["Book", summary.by_book], ["Odds range", summary.by_odds]];
  return <div className="space-y-3">
    <ProfitCurve summary={summary} />
    <div className="grid gap-3 md:grid-cols-2">{tables.filter(([, rows]) => rows?.length).map(([title, rows]) => <BreakdownTable key={title} title={title} rows={rows} />)}</div>
  </div>;
}
