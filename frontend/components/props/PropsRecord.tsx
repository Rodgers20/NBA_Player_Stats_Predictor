import type { PropsRecordResponse } from "@/lib/api";
import { formatOdds } from "@/lib/prop-research.mjs";

const signed = (value: number, digits = 1) => `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
const RESULT_TONE = { win: "text-teal-400", loss: "text-danger", push: "text-text-sec" } as const;

function RecordMetric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="rounded-lg border border-ink/10 bg-ink/[0.03] p-4">
    <p className="text-xs text-text-sec">{label}</p>
    <p className="mt-1 text-2xl font-extrabold tabular-nums text-text-pri">{value}</p>
    <p className="mt-1 text-xs text-text-sec">{detail}</p>
  </div>;
}

/** Every published pick at the price it was first shown, graded the morning after against the result and the closing line. */
export function PropsRecord({ record }: { record: PropsRecordResponse }) {
  const clv = record.clv;
  const settled = record.total + (record.pushes ?? 0);
  return <section className="ql-panel p-5">
    <h2 className="ql-section-title">Model record</h2>
    <p className="mt-1 max-w-3xl text-xs text-text-sec">
      Every pick at the price it was first published{record.since ? ` since ${record.since}` : ""}, graded the morning after.
      Beating the closing line is the early sign of a real edge; the win rate needs hundreds of picks to mean much.
    </p>
    <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <RecordMetric label="Hit rate" value={settled ? `${record.pct}%` : "—"}
        detail={`${record.hit}-${record.miss}${record.pushes ? `-${record.pushes}` : ""} · ${record.pending ?? 0} waiting`} />
      <RecordMetric label="Return at published price" value={settled && record.roi_pct != null ? `${signed(record.roi_pct)}%` : "—"}
        detail={settled ? `${signed(record.units ?? 0, 2)} units over ${settled} bets` : "No graded picks yet"} />
      <RecordMetric label="Closing-line value" value={clv?.n && clv.avg_pct != null ? `${signed(clv.avg_pct, 2)}%` : "—"}
        detail={clv?.n ? `average over ${clv.n} picks with a close` : "Needs a later price to compare"} />
      <RecordMetric label="Beat the close" value={clv?.n ? `${clv.beat_pct}%` : "—"} detail="share of picks the market moved toward" />
    </div>
    {record.recent && record.recent.length > 0 && <div className="mt-5 overflow-x-auto">
      <table className="w-full min-w-[560px] text-left text-xs">
        <thead className="ql-data-label"><tr className="border-b border-ink/10">
          <th className="py-2 pr-3">Date</th><th className="pr-3">Pick</th><th className="pr-3">Price</th>
          <th className="pr-3">Result</th><th className="pr-3">Close</th><th>CLV</th>
        </tr></thead>
        <tbody>{record.recent.map(pick => <tr key={`${pick.date}-${pick.player}-${pick.stat}-${pick.direction}`} className="border-b border-ink/[0.06] last:border-0">
          <td className="py-2 pr-3 tabular-nums text-text-sec">{pick.date.slice(5)}</td>
          <td className="pr-3 text-text-pri">{pick.player} <span className="text-text-sec">{pick.direction} {pick.line} {pick.stat}</span></td>
          <td className="pr-3 font-mono tabular-nums text-text-pri">{formatOdds(pick.price) ?? pick.price}</td>
          <td className={`pr-3 font-semibold capitalize ${RESULT_TONE[pick.result]}`}>{pick.result}{pick.actual != null ? ` (${pick.actual})` : ""}</td>
          <td className="pr-3 tabular-nums text-text-sec">{pick.close_line ?? "—"}</td>
          <td className={`tabular-nums ${pick.clv_pct == null ? "text-text-sec" : pick.clv_pct > 0 ? "text-teal-400" : "text-danger"}`}>{pick.clv_pct == null ? "—" : `${signed(pick.clv_pct)}%`}</td>
        </tr>)}</tbody>
      </table>
    </div>}
  </section>;
}
