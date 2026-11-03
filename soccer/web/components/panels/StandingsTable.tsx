import { FormBadges } from "@/components/charts/FormBadges";
import { signed } from "@/lib/format";
import type { StandingRow } from "@/lib/types";

export function StandingsTable({ rows, limit }: { rows: StandingRow[]; limit?: number }) {
  const shown = limit ? rows.slice(0, limit) : rows;
  return (
    <div className="scroll-x -mx-1">
      <table className="w-full min-w-[420px] text-[12px]">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-ink-muted">
            <th className="px-1.5 py-1.5 text-left font-medium w-6">#</th>
            <th className="px-1.5 py-1.5 text-left font-medium">Team</th>
            <th className="px-1.5 py-1.5 text-right font-medium">P</th>
            <th className="px-1.5 py-1.5 text-right font-medium">W</th>
            <th className="px-1.5 py-1.5 text-right font-medium">D</th>
            <th className="px-1.5 py-1.5 text-right font-medium">L</th>
            <th className="px-1.5 py-1.5 text-right font-medium">GD</th>
            <th className="px-1.5 py-1.5 text-right font-medium">Pts</th>
            <th className="px-1.5 py-1.5 text-left font-medium">Form</th>
          </tr>
        </thead>
        <tbody className="tnum">
          {shown.map((r) => (
            <tr key={r.team} className="border-t border-stroke/60 hover:bg-elevated/60">
              <td className="px-1.5 py-2 text-ink-muted">{r.position}</td>
              <td className="px-1.5 py-2 font-medium whitespace-nowrap">{r.team}</td>
              <td className="px-1.5 py-2 text-right text-ink-secondary">{r.played}</td>
              <td className="px-1.5 py-2 text-right text-ink-secondary">{r.won}</td>
              <td className="px-1.5 py-2 text-right text-ink-secondary">{r.drawn}</td>
              <td className="px-1.5 py-2 text-right text-ink-secondary">{r.lost}</td>
              <td className="px-1.5 py-2 text-right text-ink-secondary">{signed(r.goal_difference)}</td>
              <td className="px-1.5 py-2 text-right font-semibold">{r.points}</td>
              <td className="px-1.5 py-2"><FormBadges form={r.form} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
